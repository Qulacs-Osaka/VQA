
import numpy as np
from numpy.random import *
import copy

from .base import *


class SantaQlaus(VQA_optimizer_base):
    '''
    mini batch処理を適用しないSanta。VQEなどの、学習データと言う概念がない問題用。
    '''
    def __init__(self, hpara=None, init_spec_dict=None):
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'SantaQlaus optimizing ...'
        self.optimizer_name = 'SantaQlaus'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        if init_spec_dict:
            #print('initialize in constructor')
            #print(init_spec_dict)
            self.initialize_optimization(init_spec_dict)

    def _optimizer_preprocess(self):
        '''
        とりあえず、total_shotsを使う場合だけ作ってある
        '''
        if not self.burnin_shots:
            self.burnin_shots = self.burnin_rate * self.max_shots
            self.no_given_burnin_shots = True
        else:
            self.no_given_burnin_shots = False

    def _optimizer_postprocess(self):
        if self.no_given_burnin_shots:
            #burnin_shotsを明示的に与えておらず、rateで自動上書きした場合は、リセットする。
            #次に別のburnin_rateを与えた場合に更新されるように。
            self.burnin_shots = None
        
    def set_hyper_parameters(self, hpara):
        """
        Set hyperparameters.

        Hyperparameters are set in the following order of priority:
        1. Values provided in the `hpara` dictionary.
        2. If not present in `hpara`, use the existing attribute value, if it exists.
        3. If neither `hpara` value nor attribute exists, use a provided default value.
        So, it is enough specify the hyper parameters to be set to different values in hpara.

        Args:
            hpara (dict): Dictionary of hyperparameters. Possible keys are:
                'Monitor_var_names' (bool): If True, variable names will be monitored. Default is existing attribute or False.
                'Monitor_iter' (bool): If True, iterations will be monitored. Default is existing attribute or False.
                'Monitor_interval' (int): Interval at which to monitor variables. Default is existing attribute or 10.
                'track_interval' (int): Interval at which to track variables. Default is existing attribute or 1.
                'continue_condition' (object): Object taking arguments and returning a boolean based on whether conditions are met to continue. Default is existing attribute or None.
                'max_iter' (int): Maximum number of iterations. Default is existing attribute or None.
                'max_shots' (int): Maximum number of shots. Default is existing attribute or None.
                'max_time' (float): Maximum time. Default is existing attribute or None.
                'BMA_num' (int): Number of averages for the B model. Default is existing attribute or 1.
                
                'eta_rule' (callable): Function to return the learning rate at t. If it is given, this function is used.
                                       Otherwise, the poly decreasing rule is used. Default is existing attribute or None.
                'beta_rule' (callable): Function to return beta at t. If it is given, this function is used.
                                       Otherwise, the poly increasing rule is used. Default is existing attribute or None.
                                       これら一般のhpara ruleを使う時に、あとからハイパーパラメータを知るためには、hpara_rules クラスを利用。
                'eta' (float): Eta parameter. Default is existing attribute or 'default_eta'.
                'C' (float): C parameter. Default is existing attribute or 5.
                'sgm' (float): Sigma parameter. Default is existing attribute or 'default_sgm'.
                'mu' (float): Mu parameter. Default is existing attribute or 0.99.
                'lmd' (float): Lambda parameter. Default is existing attribute or 1e-8.
                'burnin_shots' (int): Number of burn-in shots. Default is existing attribute or None.
                'min_shots_burnin' (int): Minimum number of shots per data during burn-in. Default is existing attribute or 10.
                'min_shots_refine' (int): Minimum number of shots per data during refinement. Default is existing attribute or 10.
                'clip_s_list_rate' (int): Rate for clipping s_list. Default is existing attribute or 10.
                'use_pc' (bool): If True, use the preconditioner. Default is existing attribute or True.
                'fix_G2' (bool): If True, use the fixed G2. If False, G2 equals G1. Default is existing attribute or False.
                'fixed_G2_value' (float): If 'fix_G2' is True, this value is used as G2. Default is existing attribute or 100.0.
                'approx_del_G1' (bool): If True, use the approximate gradient G1 term. Default is existing attribute or False.
                'approx_del_G2' (bool): If True, use the approximate gradient G2 term. Default is existing attribute or False.
                'add_fluct' (bool): If True, add artificial fluctuation. Default is existing attribute or False.
                'estimate_shots_thermal': ショット数を計算する関数を引数として任意に与えることができる。使い方がややこしいので、
                                            基本的にNoneでデフォルトか、文字列で用意したものを指定するが、以下の引数をとれば、selfの属性を使った関数を自由に与えられる。
                                            (instance, estimate_next_dict, min_shots_per_data, t, G2_scaling)を引数にとる必要がある。
                                             instance<-self, estimate_next_dict<-{iEval_returnの中身と同じ項目+g_tを持つ辞書}
                                             min_shots_per_data<-1データあたりの最小ショット数, t<-iteration_num, G2_scaling<-G2の各成分のスケール
                                             return: shots_list (per data point)  注)data pointに重みがある場合は未対応
            shot_time (float, optional): Overhead time for each shot. Defaults to existing attribute or 1.
            per_circuit_overhead (float, optional): Overhead time for each circuit. Defaults to existing attribute or 0.
            communication_overhead (float, optional): Communication overhead time. Defaults to existing attribute or 0.
            manual_continue_condition (bool, optional): Manual continue condition flag. Defaults to existing attribute or False.
        """
        super().set_hyper_parameters(hpara)
        ###eta
        default_values = {
            'beta_rule': None,
            'beta_refine_rule': None,
            'eta': 0.1,
            'C': 5.,
            'sgm': 0.999,
            'mu': 0.99, #ショット数推定で使う次の値の推定のための移動平均パラメータ
            'lmd': 1e-8,
            'burnin_rate': 0.8,
            'burnin_shots': None,
            'min_shots_burnin': 10,
            'min_shots_refine': 10,
            'clip_s_list_rate': 10.,
            #'n0_lower': 1e-8,
            'use_pc': True,
            'fix_G2': False,
            'fixed_G2_value': 100.,
            'approx_del_G1': False,
            'approx_del_G2': False,
            'add_fluct': False,
            'add_fluct_s_max': 15, #全成分の中のショット数のminがこれより大きいとき、add_fluctをfalseにする。
            'refine_s_list_norm_test': True, #Falseなら、そのままburninと同じように決める
            'beta_annealing_by_shots': True,
            'G1_scaling_annealing_by_shots': True,
            'use_G1_var_correct': False,
            's_list_moving_avg': False,
            'adaptive_beta_t': False,
            'take_beta_ema': False,
            'warm_up_beta': 1,
            'warm_up_mu': 0.999,
            'thermostat_rate': 1,
            's_min_avg_num': 0,
            #'G1_pc_scaling': 1, #成分依存したpreconditionerのスケーリングを追加できる。G1はsqrt(pc)なので、G1そのものの係数でなく、preconditionerのスケーリングになるようにしていることに注意。
            'G2_scaling_power': 0, #G2を、wの何乗するか。G2_scaling_funcを、'w_power'とした場合の次数            
            'G2_scaling_func': None,
            'estimate_shots_thermal': None,
            'ni_var_lower_rate': 0.1,
            'auto_beta_t': True,
            'c_t_annealing_by_shots': True,
            'use_sqrt_eta': False,
            'warm_up_iter_num': 5,
            's_list_clip_percentile': 100, #s_listを何パーセンタイルでクリップするか。外れ値を除外する方法。
            'beta_eta_scale_factor': 100, #beta_scaled_by_eta=Trueのときに使う。etaの何倍で割るか。
                                          #最終betaを5000~10000程度にするなら、経験的に100ぐらいがちょうどいい。
            'beta_scaled_by_eta': False, #Trueだと、1/eta でbetaをスケールする。そうするとショット数がetaに依存しないスケールをする
                                        #相対分散を同じ程度にするbetaのスケールをとるということになる。つまり、etaが小さいなら、同じショット数でも分散自体小さいから、
                                        #大きなbetaを同じショット数でとれるのだから、そうしようという考え方。どれぐらい大きなbetaをとれるか、というのをetaに合わせる
            'bias_uncorrection_list': ['beta_t'],
            'random_seed': None,
            's_min_refine_avg_mean': False, #refineでmin制限をつけるときに、meanでとるか(True)、minでとるか(False)
            'update_s_min_refine': False,
            'beta_max': -1, #refine でのbetaの最大値。etaでスケールする。負の値にしたら設定無し。
            'refine_beta_scale_by_eta': False,
            'refine_beta_ema': False
        }
        self._set_hpara(hpara, default_values)
        if not hpara.get('burnin_shots', None):
            self.burnin_shots = None #burnin_shotsは、burnin_rateを
        # Calculations after setting parameters
        if self.G2_scaling_func is None:
            self.G2_scaling_func = self.identity_scaling
        elif isinstance(self.G2_scaling_func, str) and self.G2_scaling_func=='w_power':
            self.G2_scaling_func = self.G2_scaling_w_power
        if self.estimate_shots_thermal is None:
            self.estimate_shots_thermal = self.default_estimate_shots_thermal            
        self._set_default_power_rule(rule_name='beta_rule', 
                                    annealing_by_shots=self.beta_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=1., 
                                    yfin=1e3, 
                                    exp=1.3, 
                                    final_shots=self.burnin_shots, 
                                    coeff=1., 
                                    offset=1)
        self._set_default_power_rule(rule_name='beta_refine_rule', 
                                    annealing_by_shots=self.beta_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=1., 
                                    yfin=1e3, 
                                    exp=1.3, 
                                    final_shots=None,
                                    init_shots=self.burnin_shots,
                                    coeff=1., 
                                    offset=1)
        self._set_default_power_rule(rule_name='G1_scaling_rule',
                                    annealing_by_shots=self.G1_scaling_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=10, 
                                    yfin=0.1, 
                                    exp=1.5, 
                                    final_shots=self.burnin_shots, 
                                    coeff=10, 
                                    offset=0,
                                    is_decreasing=True)
        self._set_default_power_rule(rule_name='c_t_rule',
                                    annealing_by_shots=self.c_t_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=10, 
                                    yfin=0.1, 
                                    exp=1.5, 
                                    final_shots=None, 
                                    coeff=10, 
                                    offset=0,
                                    is_decreasing=True)


    
    def initialize_optimization(self, init_spec_dict):
        '''
        keys of init_spec_dict:
            target_obj
            init_params
            func_for_track
            iEvaluate
        iEvaluateは、M_totを使う関係などから、target_obj と対応していないとおかしいことになる。
        target_objは、M_tot
        '''
        n_params = self._fetch_n_params(init_spec_dict)
        #print('n_params', n_params)
        if n_params is None:
            return
        extra_var = ['iEvaluate']
        init_values_dict = {
            's_list': ('full', self.min_shots_burnin),
            'u_t': np.random.randn(n_params)*np.sqrt(self.eta), #n_paramsを参照するので一応コード文字列で
            'v_t': 'zeros',
            'grad_avg': 'zeros',
            'gvar_list_avg': 'zeros',
            'g_t_avg': 'zeros',
            'alpha_t': ('full', np.sqrt(self.eta)*self.C),
            'eta_t': self.eta,
            'beta_t': self.get_annealed_value('beta_rule', t=1, s_tot=0),
            'g_t_old': 'zeros',
            'G2_true_old': 'zeros',
            'refine_flag': 0,
            'warm_up_flag': 0,            
            's_list_avg': 'zeros',
            'mu_t': self.mu,
            'n_shots_mean_hist': [],
            'beta_t_avg': 0,
            'rng': np.random.default_rng(seed=self.random_seed)
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)

    def default_estimate_shots_thermal(self, instance, estimate_next_dict, min_shots, t=None, G2_scaling=1.):
        '''
        G2_scaling は、noise のメトリック(preconditioner)の係数: G2 = G2_scaling * G1で与える
        '''
        #logger = init_logging('estimate_shots_thermal', raw_id_for_filename=True)
        gamma = estimate_next_dict['g_t']
        chi = estimate_next_dict['grad']
        xi = estimate_next_dict['gvar_list']
        vv = instance.sgm * instance.v_t + (1-instance.sgm) * np.square(chi)

        # 温度に基づいた、次のショット数の推定
        if instance.use_G1_var_correct:
            G1_var_correct = np.square(1 - 0.5 * (1. - instance.sgm) * np.square(chi) / vv)
        else:
            G1_var_correct = 1
        #T_t = 2. / (instance.eta_t * gamma * G1_var_correct * instance.beta_t + instance.lmd)
        ######
        #self.logger.debug(f'gamma: {gamma} \n \n xi_EV: {xi_EV}, \n \n chi: {chi}, \n \n G1_var_correct: {G1_var_correct} \n \n xi_EV/G2_scaling: {xi_EV/G2_scaling}')
        instance.c_t = instance.get_annealed_value('c_t_rule', t=instance.iteration_num + 1)
        if instance.use_sqrt_eta:
            eta_coeff = np.sqrt(instance.eta_t)
        else:
            eta_coeff = instance.eta_t
        if instance.auto_beta_t: # and self.iteration_num > 9:
            denom_vect = gamma * chi
            norm2 = np.inner(denom_vect,denom_vect)
            Tr_G2 = np.sum(gamma)            
            beta_t = 2. * Tr_G2 / (instance.c_t**2 * norm2 * eta_coeff)
            #beta_t moving average
            beta_ema_dict = {'beta_t': beta_t}
            beta_ema_dict = instance.estimate_next_values(beta_ema_dict, mu=0.9, method_function=instance.moving_average, t=t, bias_uncorrection_list=self.bias_uncorrection_list)
            #beta_t_max = instance.get_annealed_value('beta_rule', t=t)
            #instance.beta_t = min(beta_ema_dict['beta_t'], beta_t_max) 
            beta_t_min = instance.get_annealed_value('beta_rule', t=t)
            instance.beta_t = max(beta_ema_dict['beta_t'], beta_t_min)
            #s_whole_list_raw = xi * eta_coeff * gamma * G1_var_correct * instance.beta_t * 0.5 / G2_scaling
            #s_whole_list_raw = xi_EV * Tr_G2 * gamma / (G2_scaling * instance.c_t**2 * norm2)
            #self.logger.debug(f'norm2: {norm2} \n \n beta_t: {instance.beta_t} \n \n Tr_G2: {Tr_G2} \n\n denom_vect: {denom_vect} \n\n c_t: {instance.c_t}')
            #G2 = G1^2 にしてみる
            #G1_var_correctをなくす
            #s_whole_list_raw = xi_EV * Tr_G2 / (G2_scaling * instance.c_t * norm2)
            #s_whole_list_raw = xi_EV * Tr_G2 * G1_var_correct / (G2_scaling * instance.c_t * norm2)
            #
#             ggrad = gamma * chi
#             max_ind = np.argmax(ggrad)
#             g_imp = gamma[max_ind]
#             ggrad_imp2 = ggrad[max_ind]**2
#             instance.beta_t = 2. * g_imp / (instance.c_t**2 * ggrad_imp2 * np.sqrt(instance.eta_t))
#             s_whole_list_raw = xi_EV * g_imp * gamma / (G2_scaling * instance.c_t**2 * ggrad_imp2)
#             self.logger.debug(f'g_imp: {g_imp} \n \n beta_t: {instance.beta_t} \n \n ggrad_imp2: {ggrad_imp2} \n\n c_t: {instance.c_t}')
        else:
            if self.take_beta_ema:
                beta_ema_dict = {'beta_t': instance.beta_t}
                beta_ema_dict = instance.estimate_next_values(beta_ema_dict, mu=0.9, method_function=instance.moving_average, t=t, bias_uncorrection_list=self.bias_uncorrection_list)
                instance.beta_t = beta_ema_dict['beta_t']
            #s_whole_list_raw = xi_EV * np.sqrt(instance.eta_t) * gamma * G1_var_correct * instance.beta_t * 0.5 / G2_scaling #/ (T_t + instance.lmd)
        s_whole_list_raw = xi * eta_coeff * gamma * G1_var_correct * instance.beta_t * 0.5 / G2_scaling #/ (T_t + instance.lmd)
        instance.s_whole_list_raw = s_whole_list_raw #Monitoring用
        
        #s_whole_list = np.clip(s_whole_list_raw, None, np.median(s_whole_list_raw) * instance.clip_s_list_rate)
        s_whole_list = np.clip(s_whole_list_raw, None, np.percentile(s_whole_list_raw, instance.s_list_clip_percentile))
        s_whole_list = np.clip(s_whole_list_raw, min_shots, None)

        s_list = (np.ceil(s_whole_list)).astype(int)       

        return s_list
       
    def G2_scaling_w_power(self, instance):
        G2_scaling = np.full_like(instance.new_params, instance.new_params[-1]**instance.G2_scaling_power)
        G2_scaling[-1] = 1.
        return G2_scaling
    
    def identity_scaling(self, instance):
        return 1
        
    def optimizer_step(self, params, **kwargs):
        '''
        ミニバッチについて勾配成分全部に一貫したサンプリングを行うプロトコル。
        （rosalinのように勾配の各成分で独立なdata点サンプルをしない）
        iEvaluateは、
        grad, gvar_list
        を返す前提。つまり、MSEのようなものも、実際の推定量の分散のupper boundで計算する
        exploration stepsでは、Santaにもとづき、ショットノイズを熱ゆらぎとして用いる
        refine stepでは、
        ただし、1 stepでのgradの更新についてだけ考えていることに注意。
        '''
        #logger = init_logging('Santa_step', raw_id_for_filename=True)
        #logging.basicConfig(level=logging.DEBUG, filename='opt_step.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        #if self.beta_rule is None:
        #    self.beta_rule = beta_geometric
        t = self.iteration_num + 1
        s_list = self.s_list.copy()
        u_t = self.u_t.copy()
        v_t = self.v_t.copy()
        alpha_t = self.alpha_t.copy()
        eta_t = self.eta_t #次のLRの値
        g_t_old = self.g_t_old.copy()
        G2_true_old = self.G2_true_old.copy()
        #refine_flag = self.refine_flag
        #xi_s0_p = self.xi_s0_p
        #######
        sgm = self.sgm
        mu = self.mu
        mu_t = self.mu_t
        lmd = self.lmd        
        #tは前のステップのカウント。
        #とりあえず応急        
        self.G1_pc_scaling = self.get_annealed_value('G1_scaling_rule', t)        
        if self.s_min_avg_num >= 1:
            #s_min_avg_num==0 だとこのようなアダプティブなs_minをとらない。
            r = self.s_min_avg_num
            if self.s_min_refine_avg_mean:
                self.n_shots_mean_hist.append(np.mean(s_list))
            else:
                self.n_shots_mean_hist.append(np.min(s_list))
            if len(self.n_shots_mean_hist) > r:
                self.n_shots_mean_hist.pop(0)
            if len(self.n_shots_mean_hist) == r:
                mean_s = np.mean(self.n_shots_mean_hist)
                #self.min_shots_burnin = max(mean_s, self.min_shots_burnin) burninはこれを使わない
                self.min_shots_refine = max(mean_s, self.min_shots_refine)
        if self.update_s_min_refine:
            self.min_shots_refine = max(self.min_shots_refine, np.min(s_list))
        if self.s_list_moving_avg: #mu=0.9にしたのでちゅうい
            s_list_ema_dict = {'s_list': s_list.copy()}
            s_list_ema_dict = self.estimate_next_values(s_list_ema_dict, mu=0.9, method_function=self.moving_average, t=t, bias_uncorrection_list=None)
            s_list = np.clip(s_list_ema_dict['s_list'], self.min_shots_burnin, None).astype(int)        
        iEval_return = self.iEvaluate(params, s_list, **kwargs)
        #辞書としたのは、任意のoutputを記録できるようにするため。
        grad = iEval_return['grad']
        gvar_list = iEval_return['gvar_list']
        ### 次の値の推定をする量の定義
        estimate_next_dict = copy.deepcopy(iEval_return)        
        if self.use_pc:
            v_t = sgm * v_t + (1.0 - sgm) * np.square(grad)
            g_t = 1 / np.sqrt(lmd + np.sqrt(v_t))
            g_t *= np.sqrt(self.G1_pc_scaling)
            estimate_next_dict['g_t'] = g_t.copy()
        else:
            g_t = 1.
            g_t *= np.sqrt(self.G1_pc_scaling)
            #gamma = 1.        
        if t==1:
            g_t_old = g_t #最初はg_{t-1}をg_tで代用
            self._track_variables(locals())
        #moving averageで、ショット数計算のための分散などの次の値の推定
        if self.iteration_num + 1 > self.warm_up_iter_num:
            estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=mu_t, method_function=self.moving_average, t=self.iteration_num + 1, bias_uncorrection_list=self.bias_uncorrection_list, t_offset=self.warm_up_iter_num) ##次の値の推定        
        # bias correctionは初期だと明らかに不安定だが、後で入れようとしたのが以下
#         if np.min(self.s_list) > 30:
#             if self.warm_up_flag == 0:
#                 self.warm_up_end_t = t
#                 self.warm_up_flag = 1
#             estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=mu_t, method_function=self.moving_average, 
#                                                            t=t, bias_correction=True, t_offset=self.warm_up_end_t) ##次の値の推定(moving average as default)
#         else:
#             estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=mu_t, method_function=self.moving_average, t=t, bias_correction=False) ##次の値の推定(moving average as default)
        if 'g_t' not in estimate_next_dict:
            estimate_next_dict['g_t'] = 1.*np.sqrt(self.G1_pc_scaling)
        ####
        G1 = g_t
        G1_ = g_t_old
        
        #G2は、burninで、真のノイズのメトリックの微分を、パラメータの更新に使う場合(approx_del_G2)だけ必要。ショット数の決定には関係ない。
        #ショット数は、G2=G1として指定したゆらぎになるように決める。
        #logger.debug(f'burnin shots: {self.burnin_shots}')
        if self.total_shots[-1] < self.burnin_shots or self.refine_s_list_norm_test==2: #exploration
            beta_t = self.beta_t
            if self.total_shots[-1] >= self.burnin_shots:
                min_shots = self.min_shots_refine
                if self.refine_beta_scale_by_eta and self.refine_flag==0:
                    self.beta_scaled_by_eta = True
                    if self.refine_beta_ema:
                        self.take_beta_ema = True
                        self.beta_t_avg = self.beta_t
                    self.refine_flag=1
            else:
                min_shots = self.min_shots_burnin
            
            if self.total_shots[-1] >= self.burnin_shots or (np.min(s_list) > self.add_fluct_s_max):
                self.add_fluct = False
            #############
            G1_var_correct = np.square(1 - 0.5*g_t**4*(1.-sgm)*np.square(grad)) #g_t のvarianceからくる補正: G1の特定の形に依存していることに注意
            if self.fix_G2:
                G2 = self.fixed_G2_value #G2は、ショット数の計算で使う。
                #G2_ = G2 #G2の一つ前の値だが、G2_trueとその前の値を使うので結局使わない。
                #beta_t_vec = 2*G2/(gvar_list_mb*eta_t*G1*G1*gt_var_correct + lmd) #各成分の異なるeffectiveな逆温度 G2\neq G1の場合
            else:
                G2 = g_t
                #G2_ = g_t_old
                #beta_t_vec = 2./(gvar_list_mb*eta_t*G1*G1_var_correct + lmd) #各成分の異なるeffectiveな逆温度: G2=G1を利用
            if t==1:
                G2_true_old = G2 #最初はG2を使う
            if self.approx_del_G2 or ('G2_true' in self.Monitor_var_names) or ('G2_true' in self.track_var_names):
                #必要な時だけ、分散から推定されるpreconditioner G2の実際の値を計算する。
                G2_true = 0.5 * beta_t * eta_t * np.square(G1) * gvar_list * G1_var_correct
                # 実際のG2のずれの記録用
                if ('G2_true_dev' in self.Monitor_var_names) or ('G2_true_dev' in self.track_var_names):
                    self.G2_true_dev = G2_true - G2 #deviationの記録用なので、他で使われていない
            else:
                G2_true = G2
            ####params 更新########            
            #art_fluct = np.zeros_like(u_t)
            new_params = params + G1 * u_t * 0.5 #まず前のu_tを使ってパラメータをすすめる
            if isinstance(self.thermostat_rate, str): # and self.thermostat_rate == 'adaptive':
                thermostat_rate = 0.001/self.eta_t
            #elif isinstance(self.thermostat_rate, str):
            #    thermostat_rate = 1.
            else:
                thermostat_rate = self.thermostat_rate
            alpha_t += thermostat_rate * (u_t*u_t - eta_t/beta_t) * 0.5 #thermostat
            if (max(-alpha_t) > 5e2):
                raise Exception('exp(-alpha_t) overflows. beta_t may be too small. Smaller eta_t may resolve this problem.')
            u_next = np.exp(-alpha_t*0.5) * u_t
            u_next += - eta_t * G1 * grad
            #実際のeffective逆温度に人工的なゆらぎを加えて補う場合#######
            #print(self.add_fluct)
            if self.add_fluct:                
                self.art_fluct_coef_vec = np.clip(2.*G1*eta_t/beta_t - G1**2 * eta_t**2 * gvar_list / s_list, 0., None)
                #print(self.art_fluct_coef_vec)
                art_fluct = self.rng.normal(0., np.sqrt(self.art_fluct_coef_vec), len(u_t))
                #art_fluct = np.sqrt(art_fluct_coef_vec) * np.random.randn(len(u_t))
                u_next += art_fluct
            #u_next += art_fluct
            if self.approx_del_G1:
                u_next += eta_t / beta_t * (1. - G1_ / (G1+lmd)) / (u_t+lmd) #これは無視してもいいらしい
            if self.approx_del_G2:
                u_next += np.sqrt(eta_t) * (alpha_t - np.sqrt(eta_t) * G2_true) * (G2_true - G2_true_old) / (thermostat_rate*u_t + lmd) #G1はキャンセルされる
            u_next *= np.exp(-alpha_t*0.5)
            dev_E = u_next*u_next - eta_t / beta_t
            alpha_t += 0.5 * thermostat_rate * dev_E
            new_params += G1 * u_next * 0.5 #params更新
            self.new_params = new_params #他の関数で使う用
            
            #次のLR,betaを使うので、先に計算して保存
            t += 1
            #logger.debug(f'total_shots[-1]: {self.total_shots[-1]} \n shots_counter[0]: {self.shots_counter[0]}'
            if not hasattr(self, 'beta_moving_avg'):
                self.beta_moving_avg = beta_t
            self.eta_t = self.get_annealed_value('LR_rule', t)
            if not self.auto_beta_t:
                if self.total_shots[-1] < self.burnin_shots:
                    self.beta_t = self.get_annealed_value('beta_rule', t)
                else:
                    self.beta_t = self.get_annealed_value('beta_refine_rule', t)
            if self.beta_scaled_by_eta:
                self.beta_t /= self.eta_t*self.beta_eta_scale_factor
            if self.adaptive_beta_t:
                beta_avg_rate = np.mean(np.exp(-np.abs(dev_E*beta_t/eta_t*0.5)))
                self.beta_moving_avg = (1 - beta_avg_rate)*self.beta_moving_avg + beta_avg_rate*self.beta_t
                self.beta_t = self.beta_moving_avg
            if self.beta_t < self.warm_up_beta:
                self.mu_t = self.warm_up_mu
            else:
                self.mu_t = self.mu
            if self.total_shots[-1] >= self.burnin_shots and self.beta_max > 0:
                self.beta_t = min(self.beta_t, self.beta_max/(self.eta_t*self.beta_eta_scale_factor)) #etaでスケールした値でクリップ
            ######### 温度に基づいた、次のショット数の推定 #############################
            G2_scaling = self.G2_scaling_func(self)
            if self.iteration_num+1 > self.warm_up_iter_num:
                s_list = self.estimate_shots_thermal(self, estimate_next_dict, min_shots=min_shots, t=self.iteration_num+1, G2_scaling=G2_scaling) #一般に関数をあとから与えられるように、selfを引数にとることにした。
            else:
                self.beta_t = self.get_annealed_value('beta_rule', t)
            self.G2_true_old = G2_true
        else: #####refinement phase ##############
            #logger.debug(f'refine {t}')
            self.refine_flag = 1
            #### params更新 #######
            new_params = params + g_t * u_t * 0.5 #まず前のu_tを使ってパラメータをすすめる
            u_next = np.exp(-alpha_t*0.5) * u_t
            u_next -= eta_t * g_t * grad
            u_next *= np.exp(-alpha_t*0.5)
            new_params += G1 * u_next * 0.5 #params更新
            #debug vanilla sgd
            ######new_params = params - eta_t * grad
            t += 1
            eta_t = self.get_annealed_value('LR_rule', t)
            if self.refine_s_list_norm_test == 1:
                if isinstance(self.eta_t, (list, np.ndarray)):
                    R_list = self.eta_t[0] / self.eta_t #G1_pcの比にすべき
                else:
                    R_list = np.full_like(grad, 1.)
                #iEval_returnを渡すか、estimate_next_dictか。前者だと、現在の値を渡す。（移動平均だと、大きいときのを引っ張って大きすぎる傾向がある?）
                s_list = self.norm_test(iEval_return, shots_norm_test_on_avg=self.shots_norm_test_on_avg,
                                                                r=self.norm_test_avg_num, gamma=self.norm_test_avg_gamma, kappa_s=self.norm_test_kappa_s, R_list=R_list,
                                                                n_min=self.min_shots_refine, eps=1e-10)
            #温度スケジュールに従ったバッチサイズとショット数を引き継ぐ 
            elif self.refine_s_list_norm_test == 0: 
                self.beta_t = self.get_annealed_value('beta_rule', t)
                s_list = self.estimate_shots_thermal(self, estimate_next_dict, self.min_shots_refine, t=t)
            #else: ##########それ以外の値なら、 burninのs_listに固定                    
        self.s_list = s_list
        self.u_t = u_next #ちゃんと更新するため、u_nextにすること注意！
        self.v_t = v_t        
        self.alpha_t = alpha_t
        self.g_t_old = g_t
        #self.refine_flag = refine_flag        
        self._print_Monitor_variables(locals())
        #print(self.alpha_t)
        self._track_variables(locals()) #_track_variablesでは、selfの値が優先されるので先に更新しないと更新した値が記録されない
        #self.xi_s0_p = xi_s0_p
        #logging.debug(f"step {t}")
        return new_params
    
class SPSA(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'SPSA optimizing ...'
        self.optimizer_name = 'SPSA'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        super().set_hyper_parameters(hpara)
        default_hpara_dict = {'c': 1e-2, 'a_rate':2.*np.pi/5., 'alpha':0.602, 'gamma':0.101}
        self._set_hpara(hpara, default_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['loss']
        init_values_dict = {}
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        c = self.c
        a_rate = self.a_rate
        alpha = self.alpha
        gamma = self.gamma
        cost_eval = self.loss
        t = self.iteration_num + 1
        s_tot = self.total_shots[-1]
        n_shots = self.get_annealed_value('n_shots_rule', t=t, s_tot=s_tot, should_floor=True)        
        delta = (-1)**np.random.binomial(n=1, p=0.5, size=len(params))
        c_t = c / t**gamma
        a_t = a_rate*c / t**alpha
        g_factor = 0.5 * cost_eval(params + c_t * delta, n_shots, **kwargs) / c_t
        g_factor -= 0.5 * cost_eval(params - c_t * delta, n_shots, **kwargs) / c_t
        grad = g_factor / delta
        new_params = params - a_t * grad
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params
    
################################################
class Adam(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'Adam optimizing ...'
        self.optimizer_name = 'Adam'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        '''
        このAdamは、とりあえずbetaたちは固定。
        '''
        super().set_hyper_parameters(hpara)
        default_hpara_dict = {'beta1':0.9, 'beta2':0.99, 'epsilon': 1e-8}
        self._set_hpara(hpara, default_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['grad']
        init_values_dict = {
            'm_t': 'zeros',
            'v_t': 'zeros'
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        #m_t / v_t のmax成分で規格化したベクトルを、n_shots_listにかけてintにしたショット数にしてみるオプションを追加
        t = self.iteration_num
        m_t = self.m_t
        v_t = self.v_t
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        s_tot = self.total_shots[-1]
        self.LR_t = self.get_annealed_value('LR_rule', t=t+1, s_tot=s_tot)
        self.s_list = self.get_annealed_value('s_list_rule', t=t+1, s_tot=s_tot, should_floor=True)
        if t==0:
            self._track_variables(locals())
        t += 1
        g = self.grad(params, self.s_list, **kwargs)
        #nt("g", g)
        self.m_t = beta1 * m_t + (1.0 - beta1) * g
        self.v_t = beta2 * v_t + (1.0 - beta2) * np.square(g)
        hat_m_t = self.m_t / (1.0 - beta1 ** t)
        hat_v_t = self.v_t / (1.0 - beta2 ** t)
        new_params = params - self.LR_t * hat_m_t / (np.sqrt(hat_v_t) + epsilon)
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params
    
class Adam_ASS(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        '''
        Adam with adaptive shots strategy (ASS) using norm test
        '''
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}        
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'Adam ASS optimizing ...'
        self.optimizer_name = 'Adam_ASS'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        '''
        このAdamは、とりあえずbetaたちは固定。
        '''
        super().set_hyper_parameters(hpara)
        default_Adam_hpara_dict = {
            'beta1':0.9,
            'beta2':0.99,
            'epsilon': 1e-8,
            'mu': 0.99, #ショット数推定で使う次の値の推定のための移動平均パラメータ
            'min_shots': 4
        }
        self._set_hpara(hpara, default_Adam_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['iEvaluate']
        init_values_dict = {
            'm_t': 'zeros',
            'v_t': 'zeros',
            'grad_avg': 'zeros',
            'gvar_list_avg': 'zeros',            
            'LR_t': self.get_annealed_value('LR_rule', t=0, s_tot=0)
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        #m_t / v_t のmax成分で規格化したベクトルを、n_shots_listにかけてintにしたショット数にしてみるオプションを追加
        t = self.iteration_num
        m_t = self.m_t
        v_t = self.v_t
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        if t==0:
            self._track_variables(locals())
        t += 1
        iEval_return = self.iEvaluate(params, self.s_list, **kwargs)
        #nt("g", g)
        grad = iEval_return['grad']
        gvar_list = iEval_return['gvar_list']
        self.m_t = beta1 * m_t + (1.0 - beta1) * grad
        self.v_t = beta2 * v_t + (1.0 - beta2) * np.square(grad)
        hat_m_t = self.m_t / (1.0 - beta1 ** t)
        hat_v_t = self.v_t / (1.0 - beta2 ** t)
        new_params = params - self.LR_t * hat_m_t / (np.sqrt(hat_v_t) + epsilon)
        s_tot = self.total_shots[-1] + self.shots_counter[0]
        self.LR_t = self.get_annealed_value('LR_rule', t=t, s_tot=s_tot) #次のLR。tはannealing_by_shotsがFalseのときだけ使われる。
        ### 次の値の推定をする量の定義
        estimate_next_dict = {
            'grad': grad,
            'gvar_list': gvar_list
        }
        estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=self.mu, method_function=self.moving_average, t=t) ##次の値の推定(moving average as default)
        if isinstance(self.LR_t, (list, np.ndarray)):
            R_list = self.LR_t[0] / self.LR_t
        else:
            R_list = np.full_like(grad, 1.)
        self.s_list = self.norm_test(estimate_next_dict, shots_norm_test_on_avg=self.shots_norm_test_on_avg,
                                                                r=self.norm_test_avg_num, gamma=self.norm_test_avg_gamma, kappa_s=self.norm_test_kappa_s, R_list=R_list,
                                                                n_min=self.min_shots, eps=1e-10)
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params
    
class gCANS(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        '''
        gCANS optimizer
        rosalin などのショット数分配は、iEvaluateの内部での処理。
        '''
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}        
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'gCANS optimizing ...'
        self.optimizer_name = 'gCANS'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)

    def set_hyper_parameters(self, hpara):
        '''
        このAdamは、とりあえずbetaたちは固定。
        '''
        #LRの_set_ruleはbaseでやっている。
        super().set_hyper_parameters(hpara)
        default_hpara_dict = {
            'L': 0.1, #Lipschitz constant
            'epsilon': 1e-8,
            'mu': 0.99, #ショット数推定で使う次の値の推定のための移動平均パラメータ
            'min_shots': 4,
            'gCANS1': True, #gain positivity checkをするかどうか(Trueならしない)
            'EMA_var': False, #sqrt{var}の次の推定として移動平均をとるときに、varの移動平均をとる(True)か、sqrtとってから平均とるか
            'bias_uncorrection_list': None
        }
        self._set_hpara(hpara, default_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['iEvaluate_iRandom']
        init_values_dict = {
            's_list': ('full', self.min_shots),
            'grad_avg': 'zeros',
            'g_sigma_list_avg': 'zeros', #EMA_var=Trueのときは、varのavgが入ることになる
            'LR_t': self.get_annealed_value('LR_rule', t=0, s_tot=0)
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        '''
        sqrt of varを使うのに、sqrtを先にとってから移動平均するのをデフォルトにしている。
        varの移動平均をとって、sqrtするのは、EMA_var=Trueにする。
        '''
        #logger = init_logging('gCANS_step', raw_id_for_filename=True)
        t = self.iteration_num
        s_list = self.s_list
        L = self.L
        #print("debug_iEval")
        if t==0:
            self._track_variables(locals())
        iEval_return = self.iEvaluate_iRandom(params, s_list, **kwargs)
        grad = iEval_return['grad']
        var_list = iEval_return['gvar_list']
        if self.gCANS1:
            #print(alpha)
            new_params = params - self.LR_t * grad
        else:
            if isinstance(self.LR_t, (list, np.ndarray)):
                alpha_list = self.LR_t
            else:
                alpha_list = np.full_like(params, self.LR_t)
            g2 = np.square(grad)
            check_list = g2/(L*(g2 + var_list/s_list + self.epsilon))
            alpha_list = np.where(alpha_list <= check_list, alpha_list, check_list)
            new_params = params - alpha_list * grad
        t += 1
        if self.EMA_var:
            estimate_next_dict = {
                'grad': grad,
                'g_sigma_list': var_list
            }
            estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=self.mu, t=t, bias_uncorrection_list=self.bias_uncorrection_list)
            #varの平均をとっているので、その平均のルートをとって、sigmaの推定とする
            estimate_next_dict['g_sigma_list'] = np.sqrt(estimate_next_dict['g_sigma_list'])
        else:
            #sigmaを先に計算して直接平均をとる。
            sgm_list = np.sqrt(var_list)
            estimate_next_dict = {
                'grad': grad,
                'g_sigma_list': sgm_list
            }
            estimate_next_dict = self.estimate_next_values(estimate_next_dict, mu=self.mu, t=t, bias_uncorrection_list=self.bias_uncorrection_list)        
        self.LR_t = self.get_annealed_value('LR_rule', t=t)
        chi = estimate_next_dict['grad']
        xi = estimate_next_dict['g_sigma_list']
        s_list = 2 * L * self.LR_t * xi * np.sum(xi) / ((2 - L * self.LR_t)*np.inner(chi,chi) + self.epsilon)
        #self.logger.info('s_list: {s_list} \n \n L: {L} \n \n LR_t: {LR_t} \n \n xi*np.sum(xi): {xi*np.sum(xi)} \n \n norm(chi)^2: {np.inner(chi,chi)}\n \n')
        s_list = np.clip(s_list, self.min_shots, None)
        s_list = (np.ceil(s_list)).astype(int)
        self.s_list = s_list
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params
    
class SGD(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'vanilla SGD optimizing ...'
        self.optimizer_name = 'vanilla SGD'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        '''
        デフォルトのみ
        '''
        super().set_hyper_parameters(hpara)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['grad']
        init_values_dict = {}
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        #m_t / v_t のmax成分で規格化したベクトルを、n_shots_listにかけてintにしたショット数にしてみるオプションを追加
        t = self.iteration_num + 1
        s_tot = self.total_shots[-1]
        self.LR_t = self.get_annealed_value('LR_rule', t=t, s_tot=s_tot)
        self.s_list = self.get_annealed_value('s_list_rule', t=t, s_tot=s_tot, should_floor=True)
        if t==1:
            self._track_variables(locals())
        g = self.grad(params, self.s_list, **kwargs)
        new_params = params - self.LR_t * g
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params

class SGDM(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'SGDM optimizing ...'
        self.optimizer_name = 'SGDM'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        '''
        とりあえずmuは固定。
        '''
        super().set_hyper_parameters(hpara)
        default_hpara_dict = {'mu':0.9}
        self._set_hpara(hpara, default_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['grad']
        init_values_dict = {
            'm_t': 'zeros'
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        #m_t / v_t のmax成分で規格化したベクトルを、n_shots_listにかけてintにしたショット数にしてみるオプションを追加
        t = self.iteration_num        
        mu = self.mu
        s_tot = self.total_shots[-1]
        self.LR_t = self.get_annealed_value('LR_rule', t=t+1, s_tot=s_tot)
        self.s_list = self.get_annealed_value('s_list_rule', t=t+1, s_tot=s_tot, should_floor=True)
        if t==0:
            self._track_variables(locals())
        t += 1
        g = self.grad(params, self.s_list, **kwargs)
        #nt("g", g)
        self.m_t = mu * self.m_t + g        
        new_params = params - self.LR_t * self.m_t
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params

class MCMC_VQA(VQA_optimizer_base):
    def __init__(self, hpara=None, init_spec_dict=None):
        '''
        MCMC enhanced VQA
        (incorporating MH sampling)
        '''
        if hpara is None:
            hpara = {}
        if init_spec_dict is None:
            init_spec_dict = {}
        super().__init__() #results_listを初期化
        ### hyper params
        self.starting_message = 'MCMC-VQA optimizing ...'
        self.optimizer_name = 'MCMC-VQA'
        self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        self.initialize_optimization(init_spec_dict)
    
    def set_hyper_parameters(self, hpara):
        '''
        beta
        xi
        burnin_rate
        '''
        super().set_hyper_parameters(hpara)
        default_hpara_dict = {
            'beta': 0.2,
            'xi': 0.5,
            'burnin_rate': 0.8,
            'random_seed': None
        }
        self._set_hpara(hpara, default_hpara_dict)
        
    def initialize_optimization(self, init_spec_dict):
        extra_var = ['loss', 'grad', 'iEvaluate']
        init_values_dict = {
            'rng': np.random.default_rng(seed=self.random_seed),
            'old_g': None,
            'old_var': None,
            'old_loss': None
        }
        self._basic_init_routine(init_spec_dict, extra_var, init_values_dict)
    
    def optimizer_step(self, params, **kwargs):
        #m_t / v_t のmax成分で規格化したベクトルを、n_shots_listにかけてintにしたショット数にしてみるオプションを追加
        t = self.iteration_num + 1
        s_tot = self.total_shots[-1]
        self.LR_t = self.get_annealed_value('LR_rule', t=t, s_tot=s_tot)
        self.s_list = self.get_annealed_value('s_list_rule', t=t, s_tot=s_tot, should_floor=True)
        if t==1:
            self._track_variables(locals())                
        if s_tot < self.max_shots*self.burnin_rate:
            self.n_shots = self.get_annealed_value('n_shots_rule', t=t, s_tot=s_tot, should_floor=True)
            if self.old_g is None:
                iEval_return = self.iEvaluate(params, self.s_list, **kwargs)
                self.old_g = iEval_return['grad']
                self.old_var = iEval_return['gvar_list']
                self.P_old = np.exp(-self.beta*self.loss(params, self.n_shots, **kwargs))
            prop_params = params - self.LR_t * self.old_g + self.xi*self.rng.normal(0, 1., len(params))            
            iEval_return = self.iEvaluate(prop_params, self.s_list, **kwargs)
            prop_g = iEval_return['grad']
            prop_var = iEval_return['gvar_list']
            prop_loss = self.loss(prop_params, self.n_shots, **kwargs)
            P_prop = np.exp(-self.beta*prop_loss)
            G_prop_old = prod_normal_pdfs(params - prop_params, self.LR_t*self.old_g, self.xi**2 + self.LR_t**2*self.old_var/self.s_list)
            G_old_prop = prod_normal_pdfs(prop_params - params, self.LR_t*prop_g, self.xi**2 + self.LR_t**2*prop_var/self.s_list)
            A_prop_old = min(1, P_prop / self.P_old * G_old_prop / G_prop_old) #acceptance rate
            if A_prop_old==1 or self.rng.uniform() <= A_prop_old:
                self.accept = 1 #記録用
                new_params = prop_params
                self.old_g = prop_g
                self.old_var = prop_var
                self.P_old = P_prop
            else:
                self.accept = 0
                new_params = params
        else:            
            new_params = params - self.LR_t * self.old_g
            self.old_g = self.grad(new_params, self.s_list, **kwargs)
        self._print_Monitor_variables(locals())
        self._track_variables(locals())
        return new_params
    
def univariate_normal_pdf(x, mu, var):
    """Computes the vector of univariate normal PDFs."""    
    exponent = -0.5 * ((x - mu) ** 2) / var
    return np.exp(exponent) / np.sqrt(2.0 * np.pi * var)

def prod_normal_pdfs(x, mu, var):
    """Computes the product of univariate normal PDFs for each component at x."""
    pdfs = univariate_normal_pdf(x, mu, var)
    return np.prod(pdfs)