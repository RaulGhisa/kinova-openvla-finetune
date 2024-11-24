import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import dm_env
from dm_env import specs
import numpy as np
import random
import cv2
from cv_bridge import CvBridge
 
from sensor_msgs.msg import JointState
# from tf2_msgs.msg import TFMessage
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
 
 
class Gen3LiteEnv(dm_env.Environment):
    def __init__(self):
        # Initialize the ROS 2 Node
        self.node = Node("data_collection_node")
 
        cb_group_1 = MutuallyExclusiveCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()
 
        # Create subscription
        self.joint_sub = self.node.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10, callback_group=cb_group_1
        )
        # self.tf_sub = self.node.create_subscription(
        #     TFMessage, "/tf", self.tf_callback, 10
        # )
 
        self.fk_client = self.node.create_client(GetPositionFK, '/compute_fk', callback_group=cb_group_2)
        # while not self.fk_client.wait_for_service(timeout_sec=1.0):
        #     print('Service not available, waiting again...')
        
        self.cap = cv2.VideoCapture(6)  # Open the default camera  #v4l2-ctl --list-devices
        self.bridge = CvBridge()
 
        self.image = None
        self.instructions = ["pick up a banana", "get a banana"] # Use self.done to judge if it should be changed
 
        # Initialize environment state
        self.current_observation = None
        self.current_reward = 0.0
        self.done = False

        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8)
        self.image = resized_dummy_image
 
    def joint_callback(self, msg):
        joint_states = msg
        joint_names = msg.name
        joint_positions = np.array(msg.position)
        # d: double-precision floating-point numbers (8 bytes per element), matching the float64 data type.
        
        # print(f"Received joint names: {joint_names}")

        joint_states_dummy = JointState()
        positions = [0.1, -0.3, 0.12, 0.1, 0.5, 0.2]
        print(f"Received joint positions: {joint_positions}")
        
        # Store in observation
        self.current_observation = {
            "joint_positions": joint_positions,
        }
 
        self.get_end_effecter_position(joint_states)
 
    def get_end_effecter_position(self, joint_states):
        print("Checking if /compute_fk service is available...")
        while not self.fk_client.wait_for_service(timeout_sec=3.0):
            print('Service not available, waiting again...')
 
        print("Preparing /compute_fk service request...")
        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = 'world'
        fk_request.fk_link_names = ['end_effector_link']
        # fk_request.robot_state.joint_state = joint_states
        fk_request.robot_state.joint_state.name = joint_states.name
        fk_request.robot_state.joint_state.position = joint_states.position
 
        print("Sending request to /compute_fk service...")
        fk_future = self.fk_client.call_async(fk_request)
        # rclpy.spin_until_future_complete(self, fk_future)
        print("Waiting for /compute_fk response...")
 
        try:
            response = fk_future.result()
            print("response ", response)
            pose_stamped = response.pose_stamped[0]
            ef_position = pose_stamped.pose.position
            print(f"End effector position: x={ef_position.x}, y={ef_position.y}, z={ef_position.z}")
        except Exception as e:
            print('Service call failed')
 
            x, y, z = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
            ef_position_dummy = Point()
            ef_position_dummy.x = x
            ef_position_dummy.y = y
            ef_position_dummy.z = z
            print("ef position dummy ", ef_position_dummy)
 
    # Constantly get image data, same as inference
    def get_camera_image(self):
        # Capture a frame from the camera
        ret, frame = self.cap.read() #frame, (1080, 1920)
        if ret:
            # Convert the frame to the expected format if necessary
            cropped_image = frame[0:1080,400:1920]
            resized_image = cv2.resize(cropped_image, (256, 256))
            self.image = resized_image
            # print("self.image ", self.image.shape[:2])
            # Render the image in a window
            # cv2.imshow("Cropped Image", cropped_image)
            cv2.imshow("Cropped Image", resized_image)
            # WaitKey allows image rendering and checks if 'q' was pressed to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()
        else:
            print("Failed to capture image")
 
 
 
    ### We need the following required abstract methods in dm_env.Environment
 
 
    def reset(self):
        # Reset the environment state
        print("Environment reset")
        self.current_observation = None
        self.current_reward = 0.0
        self.done = False
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=self.current_observation,
        )
 
    def step(self, action): # delta motion
        # Apply the action and update the environment state
        print(f"Action taken: {action}")
        self.current_reward = 1.0 # or 0.0
        self.done = self.current_reward > 10  # We need to think about it, like a switch
 
        # Change self.instruction at the end of episode using a LLM
        # Make a list of instructions and pick one of them
        if self.done:
            self.instruction = random.choice(self.instructions)
 
        return dm_env.TimeStep(
            step_type=dm_env.StepType.LAST if self.done else dm_env.StepType.MID,
            reward=self.current_reward,
            discount=0.9,
            observation=self.current_observation,
        )
 
    def action_spec(self):
        # Should be 7 values (displacement of translation, rotation and gripper)
        return specs.BoundedArray(
            shape=(1,), dtype=np.int32, minimum=0, maximum=5, name="action_spec"
        )
 
    def observation_spec(self):
        # Should be image, natural language instruction and other information
        return specs.BoundedArray(
            shape=(10,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="observation_spec"
        )
    
    # Get self.image, self.instruction and self.ef_position at the same frequency?
    def _observation(self):
        pass
 
 
 
def main(args=None):
    rclpy.init(args=args)
    gen3_lite_env = Gen3LiteEnv()
 
    executor = MultiThreadedExecutor()
    executor.add_node(gen3_lite_env.node)
    try:
        # rclpy.spin(gen3_lite_env.node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        gen3_lite_env.node.destroy_node()
        rclpy.shutdown()
 
if __name__ == "__main__":
    main()