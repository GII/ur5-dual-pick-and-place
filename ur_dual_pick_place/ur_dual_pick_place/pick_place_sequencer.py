#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# pick_place_sequencer.py
#
# Pick-and-place orchestrator for the top-grasp objects (vaca, cubo, pina, bola).
# You tell it which object to pick and it drives the whole cycle by calling the
# commander's services in the SAME order as the validated manual sequence, with
# proper failure handling. It does NOT change the commander.
#
#   - APPROACH phase (before the hand closes): if any plan fails, the whole
#     sequence is ABORTED cleanly and the arm returns to Ready_Right (nothing held).
#   - COMMITTED phase (object physically in hand): if a plan fails, a RECOVERY
#     carries the object back to its original pre-grasp, releases it there, and
#     returns to Ready_Right.
#
# Faithful to the proven manual flow:
#   - NO clear_octomap_around_object (the octomap_input_filter already removes the
#     target object from the OctoMap upstream).
#   - freeze/unfreeze are "primed" by calling the opposite first, to resync the
#     commander's frozen flag with the real camera state.
#   - descend/lift use offset_test_dz + plan_offset_test (no commander changes).
#   - ATTACH after the lift; DETACH before the placing descent.
#
# Trigger a cycle:
#   ros2 topic pub --once /ur_dual/pick_object std_msgs/msg/String "{data: 'vaca'}"
# -----------------------------------------------------------------------------

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

VALID_OBJECTS = ["vaca", "cubo", "pina", "bola"]


class PickPlaceSequencer(Node):
    def __init__(self):
        super().__init__("pick_place_sequencer")
        self.cb = ReentrantCallbackGroup()
        self._busy = False
        self._latest_class = None

        # Relative-Z offsets for descend / lift (same values as the manual flow).
        self.declare_parameter("descend_dz", -0.03)
        self.declare_parameter("lift_dz", 0.04)
        self.declare_parameter("hand_settle_s", 3.0)
        self.declare_parameter("detection_timeout_s", 10.0)
        # Small pause between a successful plan and its execution (the human
        # naturally waits here; helps the planning scene settle).
        self.declare_parameter("pre_execute_s", 0.4)
        # Settle before a relative-Z move so the current tool pose used to build
        # it has caught up after the preceding motion (the first descend follows
        # the large pre-grasp move with no pause, which made it read a stale,
        # higher pose and move up instead of down). Raise if moves are long.
        self.declare_parameter("post_move_settle_s", 2.0)

        # Commander Trigger services used by the cycle.
        self._cli = {}
        for name in ["freeze_octomap", "unfreeze_octomap",
                     "plan_pregrasp_from_latest_pose", "plan_offset_test",
                     "close_hand", "open_hand", "attach_grasped_object",
                     "detach_grasped_object", "plan_ready_right",
                     "plan_place_normal"]:
            self._cli[name] = self.create_client(
                Trigger, f"/ur_dual/{name}", callback_group=self.cb)
        self._exec_cli = self.create_client(
            SetBool, "/ur_dual/execute_last_plan", callback_group=self.cb)

        # Parameter services: bridge target_class, and commander offset_test_dz.
        self._bridge_set = self.create_client(
            SetParameters, "/object_pose_bridge/set_parameters", callback_group=self.cb)
        self._cmd_set = self.create_client(
            SetParameters, "/ur_dual_commander/set_parameters", callback_group=self.cb)
        self._cmd_get = self.create_client(
            GetParameters, "/ur_dual_commander/get_parameters", callback_group=self.cb)

        self.create_subscription(
            String, "/ur_dual/object_class", self._on_class, 10, callback_group=self.cb)
        self.create_subscription(
            String, "/ur_dual/pick_object", self._on_pick, 10, callback_group=self.cb)

        self.get_logger().info(
            "PickPlaceSequencer ready.\n"
            "  ros2 topic pub --once /ur_dual/pick_object std_msgs/msg/String "
            "\"{data: 'vaca'}\"\n"
            f"  Objects: {VALID_OBJECTS}")

    # ----- subscriptions -----

    def _on_class(self, msg):
        self._latest_class = msg.data

    def _on_pick(self, msg):
        obj = msg.data.strip()
        if self._busy:
            self.get_logger().warn("A sequence is already running; ignoring request.")
            return
        if obj not in VALID_OBJECTS:
            self.get_logger().error(f"'{obj}' is not valid. Options: {VALID_OBJECTS}")
            return
        threading.Thread(target=self._run, args=(obj,), daemon=True).start()

    # ----- low-level helpers -----

    def _wait_future(self, fut, label, timeout):
        start = time.monotonic()
        while rclpy.ok() and not fut.done():
            if time.monotonic() - start > timeout:
                self.get_logger().error(f"Timeout calling {label}.")
                return None
            time.sleep(0.05)
        return fut.result()

    def _trigger(self, name, timeout=180.0):
        """Call a commander Trigger service; return True on success and log the
        commander's message when it fails."""
        cli = self._cli[name]
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"/ur_dual/{name} not available.")
            return False
        res = self._wait_future(cli.call_async(Trigger.Request()), name, timeout)
        ok = res is not None and res.success
        if ok:
            self.get_logger().info(f"{name}: OK")
        else:
            self.get_logger().error(
                f"{name}: FAIL ({res.message if res else 'no response'})")
        return ok

    def _execute(self, timeout=180.0):
        """Execute the commander's pending plan; log its message on failure."""
        if not self._exec_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("execute_last_plan not available.")
            return False
        req = SetBool.Request(); req.data = True
        res = self._wait_future(self._exec_cli.call_async(req),
                                "execute_last_plan", timeout)
        ok = res is not None and res.success
        if ok:
            self.get_logger().info("execute_last_plan: OK")
        else:
            self.get_logger().error(
                f"execute_last_plan: FAIL ({res.message if res else 'no response'})")
        return ok

    def _plan_exec(self, name):
        """Plan with `name`, pause briefly, then execute. True only if both pass."""
        if not self._trigger(name):
            return False
        time.sleep(float(self.get_parameter("pre_execute_s").value))
        return self._execute()

    def _settle_hand(self):
        time.sleep(float(self.get_parameter("hand_settle_s").value))

    # ----- octomap freeze/unfreeze with priming -----
    # The commander's frozen flag can desync from the real camera when a
    # start/stop call fails. Calling the opposite service first resyncs it,
    # which is exactly what the manual sequence does.

    def _prime_freeze(self):
        self._trigger("unfreeze_octomap")        # prime (resync), ignore result
        return self._trigger("freeze_octomap")

    def _prime_unfreeze(self):
        self._trigger("freeze_octomap")          # prime (resync), ignore result
        return self._trigger("unfreeze_octomap")

    # ----- relative-Z descend / lift via offset_test (manual mechanism) -----

    def _set_offset_dz(self, dz):
        """Set offset_test_dz on the commander and verify it took (read-back), so
        we never run plan_offset_test with a stale/default value."""
        if not self._cmd_set.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("ur_dual_commander/set_parameters not available.")
            return False
        p = Parameter()
        p.name = "offset_test_dz"
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(dz))
        req = SetParameters.Request(); req.parameters = [p]
        if self._wait_future(self._cmd_set.call_async(req), "set offset_test_dz", 10.0) is None:
            return False
        # Read back to confirm the value landed on the right node.
        if not self._cmd_get.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("ur_dual_commander/get_parameters not available.")
            return False
        greq = GetParameters.Request(); greq.names = ["offset_test_dz"]
        gres = self._wait_future(self._cmd_get.call_async(greq), "get offset_test_dz", 10.0)
        if gres is None or not gres.values:
            self.get_logger().error("Could not read back offset_test_dz.")
            return False
        got = gres.values[0].double_value
        if abs(got - float(dz)) > 1e-4:
            self.get_logger().error(
                f"offset_test_dz read-back mismatch (got {got:.3f}, wanted {dz:.3f}).")
            return False
        self.get_logger().info(f"offset_test_dz set & verified = {got:.3f}")
        return True

    def _descend(self):
        # Let the pose settle after the preceding motion before reading it.
        time.sleep(float(self.get_parameter("post_move_settle_s").value))
        dz = float(self.get_parameter("descend_dz").value)
        return self._set_offset_dz(dz) and self._plan_exec("plan_offset_test")

    def _lift(self):
        time.sleep(float(self.get_parameter("post_move_settle_s").value))
        dz = float(self.get_parameter("lift_dz").value)
        return self._set_offset_dz(dz) and self._plan_exec("plan_offset_test")

    # ----- object selection -----

    def _select_object(self, obj):
        if not self._bridge_set.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("object_pose_bridge/set_parameters not available.")
            return False
        p = Parameter()
        p.name = "target_class"
        p.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=obj)
        req = SetParameters.Request(); req.parameters = [p]
        if self._wait_future(self._bridge_set.call_async(req), "set target_class", 10.0) is None:
            return False
        timeout = float(self.get_parameter("detection_timeout_s").value)
        start = time.monotonic()
        while rclpy.ok() and time.monotonic() - start < timeout:
            if self._latest_class == obj:
                self.get_logger().info(f"Detection of '{obj}' confirmed.")
                return True
            time.sleep(0.1)
        self.get_logger().error(f"No '{obj}' detection within {timeout:.0f}s.")
        return False

    # ----- failure handlers -----

    def _abort_clean(self, reason):
        """Approach-phase failure: nothing held. Return to Ready_Right, unfreeze."""
        self.get_logger().error(f"ABORT (approach): {reason}. Returning to Ready_Right.")
        self._plan_exec("plan_ready_right")   # best effort
        self._prime_unfreeze()
        return False

    def _recover_with_object(self, reason):
        """Committed-phase failure: object held. Carry it back to its original
        pre-grasp and release it there, then return to Ready_Right. Falls back to
        releasing in place if a recovery plan also fails."""
        self.get_logger().error(
            f"RECOVERY (committed): {reason}. Carrying object back to its pick pose.")
        if self._plan_exec("plan_ready_right") and \
           self._plan_exec("plan_pregrasp_from_latest_pose") and \
           self._descend():
            self._trigger("detach_grasped_object")
            self._trigger("open_hand"); self._settle_hand()
            self._lift()
        else:
            self.get_logger().error("Recovery path blocked; releasing in place.")
            self._trigger("detach_grasped_object")
            self._trigger("open_hand"); self._settle_hand()
        self._plan_exec("plan_ready_right")   # best effort
        self._prime_unfreeze()
        return False

    # ----- main sequence (mirrors the validated manual flow) -----

    def _run(self, obj):
        self._busy = True
        try:
            self.get_logger().info(f"=== Pick-and-place: '{obj}' ===")

            if not self._select_object(obj):
                return

            # --- APPROACH (not holding) -> clean abort on any failure ---
            if not self._prime_freeze():
                return self._abort_clean("freeze_octomap failed")
            if not self._plan_exec("plan_pregrasp_from_latest_pose"):
                return self._abort_clean("pre-grasp plan failed")
            if not self._descend():
                return self._abort_clean("descend onto object failed")

            # Close the hand: from here the object is physically held.
            self._trigger("close_hand"); self._settle_hand()

            # --- COMMITTED (holding) -> recovery on any failure ---
            # Lift FIRST, then attach (attaching while down collides with the table).
            if not self._lift():
                return self._recover_with_object("lift failed")
            self._trigger("attach_grasped_object")
            if not self._plan_exec("plan_ready_right"):
                return self._recover_with_object("ready (with object) failed")
            if not self._plan_exec("plan_place_normal"):
                return self._recover_with_object("place plan failed")

            # --- PLACE: detach BEFORE the descent, then lower, then release ---
            self._trigger("detach_grasped_object")
            if not self._descend():
                return self._recover_with_object("place descent failed")
            self._trigger("open_hand"); self._settle_hand()

            # --- RETREAT + reset (object already released) ---
            if not self._lift():
                return self._abort_clean("retreat failed")
            self._plan_exec("plan_ready_right")   # best effort
            self._prime_unfreeze()

            self.get_logger().info(f"=== '{obj}' COMPLETE ===")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = PickPlaceSequencer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()