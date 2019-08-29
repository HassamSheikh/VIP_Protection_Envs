import numpy as np
from pyglet.window import key
import rospy
from sensor_msgs.msg import Joy

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 0.1
            if self.move[1]: u[2] += 0.1
            if self.move[3]: u[3] += 0.1
            if self.move[2]: u[4] += 0.1
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[1] = True
        if k==key.RIGHT: self.move[0] = True
        if k==key.UP:    self.move[3] = True
        if k==key.DOWN:  self.move[2] = True

    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[1] = False
        if k==key.RIGHT: self.move[0] = False
        if k==key.UP:    self.move[3] = False
        if k==key.DOWN:  self.move[2] = False

class PSController(Policy):
    def __init__(self, env, agent_index, button_control=False):
        super(PSController, self).__init__()
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("joy", Joy, self.key_press)
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        self.button_control=button_control

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 0.03
            if self.move[1]: u[2] += 0.03
            if self.move[2]: u[4] += 0.03
            if self.move[3]: u[3] += 0.03
            if True not in self.move:
                u[0] += 1.0
        print(u)
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    def _axes_to_movement(self, axes):
        if axes[0] > 0:
            self.move[1] = True
            self.move[0] = False
        elif axes[0] < 0:
            self.move[0] = True
            self.move[1] = False
        if axes[1] > 0:
            self.move[3] = True
            self.move[2] = False
        elif axes[1] < 0:
            self.move[2] = True
            self.move[3] = False
        if sum(axes) == 0:
            self.move = [False for i in range(4)]
    def _button_to_movement(self, axes):



    def key_press(self, message):
        if not self.button_control:
            axes = message.axes[:2]
            self._axes_to_movement(axes)
        else:
            buttons = message.buttons
            self.move[0] = buttons[-1] # Index -1 represents right movement
            self.move[1] = buttons[-2] # Index -2 represents left movement
            self.move[2] = buttons[-3] # Index -3 represents down movement
            self.move[3] = buttons[-4]# Index -4 represents up movement
