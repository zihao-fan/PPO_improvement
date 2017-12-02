import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.ddpg.memory import Memory
from baselines.ddpg.ddpg import DDPG
from models import Actor, Critic

use_ddpg = True
use_annealing = False
if not use_ddpg:
    use_annealing = False

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm,
                # ddpg related params
                layer_norm=False, tau=0.001, normalize_returns=False, normalize_observations=True,
                batch_size=128, critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, popart=False, clip_norm=10., reward_scale=1.):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
        
        # init DDPG
        critic = Critic(layer_norm=layer_norm)
        actor = Actor(ac_space.shape[-1], layer_norm=layer_norm)
        memory = Memory(limit=int(1e6), action_shape=ac_space.shape, observation_shape=ob_space.shape)
        ddpg_agent = DDPG(actor, critic, memory, ob_space.shape, ac_space.shape,
                        gamma=0.99, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
                        batch_size=batch_size, action_noise=None, param_noise=None, critic_l2_reg=critic_l2_reg,
                        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)

        ddpg_agent.initialize(sess)
        ddpg_agent.reset()

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        if use_annealing:
            DDPG_AC = tf.placeholder(tf.float32, (None,)+ac_space.shape)
            DDPG_W = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        if use_annealing:
            pi_mean = train_model.pi
            ac_loss = tf.reduce_mean(tf.square(pi_mean - DDPG_AC))
        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        # ----------------- DDPG -----------------
        if use_ddpg:
            loss = pg_loss - entropy * ent_coef
            if use_annealing:
                loss = pg_loss - entropy * ent_coef + ac_loss * DDPG_W
        else:
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        # ----------------- DDPG -----------------
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None, ddpg_acs=None, ddpg_w=0.):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            if not use_annealing:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, 
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            else:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, 
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,
                        DDPG_AC:ddpg_acs, DDPG_W:ddpg_w}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            if not use_annealing:
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                    td_map
                )[:-1]
            else:
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, ac_loss, _train],
                    td_map
                )[:-1]
        if not use_annealing:
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        else:
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'ac_loss']
        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.agent = ddpg_agent
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps): # for iteration
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            # ----------------- DDPG -----------------
            if use_ddpg:
                _, q = self.model.agent.pi(self.obs[0], apply_noise=False, compute_Q=True)
                values = q[:, 0]
            # ----------------- DDPG -----------------
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            old_obs = self.obs.copy()[:]
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            # ------------------ save transitions ------------------ 
            if use_ddpg:
                self.model.agent.store_transition(old_obs, actions[:], rewards, self.obs[:].copy(), self.dones.copy())
            # ------------------ save transitions ------------------ 
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones) # original
        # ----------------- DDPG -----------------
        if use_ddpg:
            _, my_values = self.model.agent.pi(self.obs[0], apply_noise=False, compute_Q=True)
            last_values = my_values[:, 0]
        # ----------------- DDPG -----------------
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs)), 
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=16, noptepochs=4, cliprange=0.2,
            save_interval=0, nddpgbatches=32, ddpg_per_ppo=128, target_lag=1,
            ddpg_ac_weight=0.1, annealing_updates=50, with_ddpg=True, with_annealing=True):
    global use_ddpg
    global use_annealing
    use_ddpg = with_ddpg
    use_annealing = with_annealing
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, batch_size=nddpgbatches)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    print('nupdates', nupdates)
    ddpg_w = ddpg_ac_weight if use_annealing else 0.0
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        if ddpg_w > 0.0:
            ddpg_w -= 1/float(annealing_updates) * ddpg_w
        values_list = []
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, rewards, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if use_annealing:
            ddpg_ac_list = []
            for idx in range(obs.shape[0]):
                ddpg_ac, _ = model.agent.pi(obs[idx], apply_noise=False, compute_Q=False)
                ddpg_ac_list.append(ddpg_ac)
            ddpg_ac = np.asarray(ddpg_ac_list)
        values_list.append(values)
        # print('obs.shape', obs.shape, 'rewards.shape', returns.shape, 'masks.shape', masks.shape, 'actions.shape', actions.shape)
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    if not use_annealing:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                    else:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, ddpg_acs=ddpg_ac[mbinds], ddpg_w=ddpg_w))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    # mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                    if not use_annealing:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                    else:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates, ddpg_acs=ddpg_ac[mbinds], ddpg_w=ddpg_w))        

        if use_ddpg:
            mbcritic_loss = []
            mbactor_loss = []
            # ------------- train DDPG ----------------
            for _ in range(ddpg_per_ppo * noptepochs * nminibatches):
                cl, al = model.agent.train()
                mbcritic_loss.append(cl)
                mbactor_loss.append(al)
                if update > target_lag:
                    model.agent.update_target_net()
            # print('noptepochs', noptepochs, 'nbatch_train', nbatch_train, 'nbatch', nbatch)
            # ------------- train DDPG ----------------

        lossvals = np.mean(mblossvals, axis=0)
        values_avg = np.mean(values_list)
        if use_ddpg:    
            critic_loss = np.mean(mbcritic_loss)
            actor_loss = np.mean(mbactor_loss)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            logger.logkv('value estimation', values_avg)
            logger.logkv('eprew_max', np.max(mblossvals))
            logger.logkv('eprew_min', np.min(mblossvals))
            logger.logkv('eprew_std', np.std(mblossvals))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if use_ddpg:
                logger.logkv('critic_loss', critic_loss)
                logger.logkv('actor_loss', actor_loss)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
