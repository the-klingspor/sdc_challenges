from pyvirtualdisplay import Display
import gym
import deepq
import argparse
import platform
import time

def load_actions ( action_filename ):

    actions = []

    with open ( action_filename ) as f:

        lines = f.readlines()

        for line in lines:
            action = []
            for tok in line.split():
                action.append ( float ( tok ))
            actions.append (action)

    return actions

def main ():

    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """ 

    print("python version:\t{0}".format (platform.python_version()))
    print("gym version:\t{0}".format(gym.__version__))

    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--total_timesteps', type=int, default=100000, help='The number of env steps to take')
    parser.add_argument('--action_repeats', type=int, default=3, help='Update the model every action_repeats steps')
    parser.add_argument('--gamma', type=float, default=0.95, help='selection action on every n-th frame and repeat action for intermediate frames')
    parser.add_argument('--exploration_fraction', type=float, default=1.0, help='amount of time to use epsilon greedy exploration')
    parser.add_argument('--action_filename', type=str, default='five_actions.txt', help='a list of actions')
    parser.add_argument('--use_doubleqlearning', default=False, action="store_true", help='a flag that indicates the use of double q learning')
    parser.add_argument('--use_ema', default=False, action="store_true", help='a flag that indicates the use of an exponential moving average for the target network')
    parser.add_argument('--gas_schedule', default=False, action="store_true", help='a flag that indicates the use of a linear schedule for prioritized gas actions')
    parser.add_argument('--display', default=False, action="store_true", help='a flag indicating whether training runs in the cluster')
    parser.add_argument('--agent_name', type=str, default='agent', help='an agent name')
    parser.add_argument('--outdir', type=str, default='', help='a directory for output')
    parser.add_argument('--buffer_size', type=int, default=100000, help='buffer size')
    parser.add_argument('--PRB', default=False, action='store_true', help='a flag to enable th prioritized replay buffer')

    args = parser.parse_args()

    # check args
    print("\nArgs information")
    print("total_timesteps:     {0}".format(args.total_timesteps))
    print("action_repeats:      {0}".format(args.action_repeats))
    print("gamma:               {0}".format(args.gamma))
    print("exploration_fraction {0}".format(args.exploration_fraction))
    print("action_filename:     {0}".format(args.action_filename))
    print("use_doubleqlearning: {0}".format("doubleq" if args.use_doubleqlearning else "no doubleq"))
    print("use_ema:             {0}".format("ema" if args.use_ema else "no ema"))
    print("gas_schedule:        {0}".format("true" if args.gas_schedule else "false"))
    print("display:             {0}".format("true" if args.display else "false"))
    print("agent_name:          {0}".format(args.agent_name))
    print("outdir:              {0}".format(args.outdir))
    print("buffer_size:         {0}".format(args.buffer_size))
    print("prioritized replay buffer (PRB): {0}".format(args.PRB))

    # load a virtual display if we run training in the cluster that has no main display.
    if not args.display:
        display = Display(visible=0, size=(800, 600))
        display.start()

    # load actions
    actions = load_actions(args.action_filename)
    print("actions:\t\t", actions)

    # start training
    print("\nStart training...")
    env = gym.make("CarRacing-v0")

    deepq.learn(env, total_timesteps=args.total_timesteps,
                action_repeat=args.action_repeats,
                gamma=args.gamma,
                exploration_fraction=args.exploration_fraction,
                model_identifier=args.agent_name,
                outdir=args.outdir,
                new_actions=actions,
                use_doubleqlearning=args.use_doubleqlearning,
                use_ema=args.use_ema,
                gas_schedule=args.gas_schedule,
                buffer_size=args.buffer_size,
                PRB=args.PRB
                )
    
    # wrap up
    env.close()
    if not args.display:
        display.stop()


if __name__ == '__main__':
    main()
