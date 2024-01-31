#とりあえず単純な3-class分類専用になっています。

import os
import pandas as pd
import itertools
import copy
import seaborn as sns

from VQA_optimizers import qml_optimizers
from VQA_optimizers.base import init_logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import itertools
import numpy as np
import time
from qulacs import Observable

from numpy.random import *

import matplotlib.ticker as ptick


from pathos.multiprocessing import ProcessingPool

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import inspect

from myQNNs_base import *

import matplotlib.ticker as ptick

from multiprocessing import cpu_count

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import dill
import hashlib
import uuid
from datetime import datetime


class myQNN_classifier(myQNN_base):
    '''
    とりあえず 3-class専用で未完成。
    learning_circuit : skqulacsのlearning_circuit クラスを入れる。           
    optimizer の対象となる target_obj として与えられ, clientとして使われる
    最適化に使う勾配計算などの提供や、内部カウンターなどの連携のみに使う。（並列化するときは、別のインスタンスを処理ごとにつくるなど同期に注意）
    最適化の結果の管理は別で。
    '''
    def __init__(self, n_qubits, learning_circuit,
                 x_train=None, y_train=None, x_test=None, y_test=None, min_approx_shots=-1, setting_dict=None, id_string="QNN", file_id=None,
                 obs=None, num_classes=3,
                shot_time=1., per_circuit_overhead=0., communication_overhead=0., max_sub_nqubits_sampling_aft_partial_tr=6, partial_MSE=False, eps=0.,
                class_unbias=True, noise_c_init=0.5, n_ZNE_min = 2,
                density=False, noisy=False, exact_clean=True, noise_scales=None):
        '''
        min_approx_shots: 何ショット以上はCLTで近似するか。-1とすると近似なし。
        setting_dict: optimizerで使う gradや iEval などを指定する
        noisy=True なら、learning_circuitは、自分で定義した noisyLearningCircuitでないといけない。exactでdensityを使って計算できる（density=Trueの場合）
        exact_clean: Trueなら、exactをclean circuitのexact valueで出す
        max_sub_nqubits_sampling_aft_partial_trはobservableのサンプリングをするときに、ターゲットが何qubitまでpartial traceとってからサンプリングするか
        （計算速度に関わる。大体実験したら、少数では圧倒的にpartial traceだが、全qubit数が大きい場合、半分ぐらいから微妙になってくるので、
        ## 9-qubitまでは、常にpartial trace
        ## 10-qubit では k=7 からdirect
        ## 12-qubitぐらい以上の k=5あたりからはdirect samplingでmodとるほうがよさげ。partial traceをとる時間と、部分系のdensity matrixの次元が大きくなる影響と思われる。）
        ただし今は常にpartial traceとっていて、切り替えは未実装なので今は気にしない。（observableとして Z0を前提としたものしか作ってない）
        '''
        global generate_labels_flag        
        generate_labels_flag = False
        self.noisy = noisy
        self.density = density #Bool densityを使って期待値計算するかどうか
        self.exact_clean = exact_clean #Bool exact valueをclean circuitで計算するか        
        self.partial_MSE = partial_MSE
        self.num_classes = num_classes
        self.circuit = learning_circuit
        if noisy and hasattr(learning_circuit, '_scale_factors'):
            self.noise_scales = learning_circuit._scale_factors.copy()
            self.noise_scales[0] = 1.
        elif noise_scales is not None:
            self.noise_scales = noise_scales.copy()
            self.noise_scales[0] = 1.
        #self.noise_para = [1.] #extrapolationなどで使うノイズパラメータの推定値
        self.class_unbias = class_unbias
        if class_unbias:
            self.noise_a_vect = [1./3., 1./3., 1./3.]
        else:
            self.noise_a_vect = [0.5, 0.25, 0.25]
        self.n_ZNE_min = n_ZNE_min
        #self.sigma_rate1_2 = 1. #expZNEで使うsigma_1/sigma_2
        self.n_c_params = len(self.circuit.get_parameters())
        self.n_params = self.n_c_params #re-normalization coeffなし        
        self._eps=eps
        self.noise_c = np.full((2,self.n_params), noise_c_init) #param-shift0,1
        self.noise_c_no_shift = noise_c_init
        self.sgm_12 = np.full((2,self.n_params), 1.) #param-shift0,1
        self.sgm_12_no_shift = 1.
        self.QEM_time = np.full((2,self.n_params), 0)
        self.QEM_time_no_shift = 0
        self.EMA_para = 0.9
        #num_classes = max(np.concatenate((y_train, y_test))) + 1
        #self.num_classes = num_classes
        self.set_data(x_train, y_train, x_test, y_test)
        self.methods_for_optimization = {}
        if setting_dict:
            self.set_methods_for_optimizer(setting_dict)
        #self.process_id = process_id
        #self.y_test_1hot = np.array([label2one_hot(y_test[i], num_classes) for i in range(len(y_test))])
        self.min_approx_shots = min_approx_shots
        self.call_counter = [0] #1data点のpredictの呼び出し回数。predict一回に複数回路を使う場合、circuit_counterとずれる。
        self.shots_counter = [0]
        self.circuit_counter = [0]
        self.epoch_counter = [0]
        self.outcomes_cache = [[]]
        self.n_qubits = n_qubits
        self.communication_overhead = communication_overhead
        self.shot_time = shot_time
        self.per_circuit_overhead = per_circuit_overhead
        self.id_string = id_string
        self.results_dict = {}
        #self._uniform_wt = np.full(len(self.x_train), 1.) #計算されるショット数は、各データ点への割当が直接出るようにしたので、１
        #self._uniform_wt_dist = np.full(len(self.x_train), 1.) / len(self.x_train)
        #self._uniform_wt = np.full(len(self.x_train), 1. / len(self.x_train))
        if not file_id:
            self.file_id = str(uuid.uuid4())
        else:
            self.file_id = file_id
        self.exact_cost_evol = []
        self.remain_ind = [] #mini_batchやるため
        print("file_id:", self.file_id)
        self._dim = 2**self.n_qubits
        #if obs is None:
        #   obs = self._default_obs()
        #self.obs_json=obs.to_json() #拡張するとき用。今は固定したものしか使わないので、これは使わない。（実行時間は読み込みのほうが遅い）
        ###これらは、一般のオブザバブルで推定する場合に使う。（とりあえず未実装で、Z0だけ作った）
        # self.target_indices_list = []
        # for i in range(obs.get_term_count()):
        #     term = obs.get_term(i)
        #     self.target_indices_list.append(term.get_index_list())
        self.max_sub_nqubits_sampling_aft_partial_tr = max_sub_nqubits_sampling_aft_partial_tr #obsのsamplingするときに、target以外のpartial traceとるか、直接やってmodとるか
        ###
        self.n_shots_test = 1e2 #test accuracyをsamplingでevaluateするときに各dataに使うショット数

    def get_parameter_count(self, return_circuit_parameter_count=False):
        if return_circuit_parameter_count:
            return self.n_c_params
        else:
            return self.n_params

    def set_data(self, x_train, y_train, x_test, y_test):
        if (not (x_train is None)) and (not (y_train is None)):
            if self.num_classes is None:
                self.num_classes = np.max(y_train) + 1
            else:
                self.num_classes = np.maximum(self.num_classes, np.max(y_train) + 1)
            num_bit = len(bin(self.num_classes - 1)) - 2
            self.num_bit = num_bit
            if num_bit==1:
                def parity(x):
                    '''
                    クラスindexを表すための最小ビット数だけを見る、ローカルな測定
                    '''
                    return x % self.num_classes
            else:
                mod2n = 2 ** num_bit
                def parity(x):
                    '''
                    クラスindexを表すための最小ビット数だけを見る、ローカルな測定
                    '''
                    return (x % mod2n) % self.num_classes
            self.interpret = parity
            self.x_train = np.array(x_train)
            self.y_train = np.array(y_train)
            self.num_train_data = len(x_train)
            self._uniform_wt = np.full(len(self.x_train), 1.) #計算されるショット数は、各データ点への割当が直接出るようにしたので、１
            self._uniform_wt_dist = np.full(len(self.x_train), 1.) / len(self.x_train)
            self.y_train_1hot = np.array([label2one_hot(y_train[i], self.num_classes) for i in range(len(y_train))])
            #self.y_train_1hot = np.array([label2one_hot(y_train[i], num_classes) for i in range(len(y_train))])
        if (not (x_test is None)) and (not (y_test is None)):
            if self.num_classes is None:
                self.num_classes = np.max(y_test) + 1
            else:
                self.num_classes = np.maximum(self.num_classes, np.max(y_test) + 1)
            num_bit = len(bin(self.num_classes - 1)) - 2
            self.num_bit = num_bit
            if num_bit==1:
                def parity(x):
                    '''
                    クラスindexを表すための最小ビット数だけを見る、ローカルな測定
                    '''
                    return x % self.num_classes
            else:
                mod2n = 2 ** num_bit
                def parity(x):
                    '''
                    クラスindexを表すための最小ビット数だけを見る、ローカルな測定
                    '''
                    return (x % mod2n) % self.num_classes
            self.interpret = parity #samplesをとって、クラスをインデックスに変換する関数
            self.x_test = np.array(x_test)
            self.y_test = np.array(y_test)
            self.y_test_1hot = np.array([label2one_hot(y_test[i], self.num_classes) for i in range(len(y_test))])

    def compile(self, optimizer='SGD', loss='mean_squared_error', hpara=None, init_params=None, min_approx_shots=None,
                x_train=None, y_train=None, x_test=None, y_test=None,
                shot_time=None, per_circuit_overhead=None, communication_overhead=None,
                init_spec_dict=None, manual_methods_setting_dict=None, partial_MSE=False, eps=0.):
        '''
        *** これはベースクラスで部分的に定義できると思われる
        optimizerとloss function を設定する。
        基本的には、optimizerと、hparaを与えれば良い。lossは違うものを使いたいときは指定。
        init_paramsや、hparaもあとから指定可能。
        min_approx_shotsや、各種overhead time は改めて指定できるが、変更なしなら何もしなくて良い。

        loss: loss functionをいろいろ用意する場合は、これで指定したのに合わせて、optimizerで使用するgradなどを設定する。
        （既知のoptimizerに対して個別の処理を実装することもできる(base classに作り直すときに関係ある)）
        hparaは辞書でそのままoptimizer_class に渡せるように与える。（最適化を走らせるときに改めて与えることもできる）
        self.optimizer に、指定した optimizer の文字列のクラスのインスタンスを保持している。
        改めてデータをセットすることもできる
        '''
        self.set_data(x_train, y_train, x_test, y_test)
        self.partial_MSE = partial_MSE        
        self._optimizer_name = optimizer
        self._eps = eps
        if min_approx_shots is not None:
            self.min_approx_shots = min_approx_shots
        optimizer_class = getattr(qml_optimizers, optimizer)
        if init_spec_dict is None:
            init_spec_dict = {}
        if init_params is not None:
            init_spec_dict['init_params'] = init_params
            self.model_params = init_params
        self.set_overhead_time(shot_time, per_circuit_overhead, communication_overhead)

        if manual_methods_setting_dict is not None:
            self.set_methods_for_optimizer(manual_methods_setting_dict)
        else:
            # 以下では、デフォルトの勾配計算などのメソッドたちを最適化用にセットする
            # どうしても別の関数を定義しないといけないoptimizerがあるときは分岐させる。
            # noisy バージョンを追加するときは、loss で分岐すればいい。
            if loss == 'mean_squared_error':
                methods_setting_dict = {
                    'func_to_track': 'exact_predict_error_and_MSE',
                    'loss': 'MSE_loss_eval',
                    'grad': 'MSE_grad_eval',
                    'iEvaluate': 'MSE_iEval',
                    'iEvaluate_iRandom': 'MSE_iEval_iRandom'
                }
                self.set_methods_for_optimizer(methods_setting_dict)
            elif loss == 'cross_entropy':
                methods_setting_dict = {
                    'func_to_track': 'exact_error_both_to_track',
                    'loss': 'CE_loss_eval',
                    'grad': 'CE_grad_eval',
                    'iEvaluate': 'CE_iEval',
                    'iEvaluate_iRandom': 'CE_iEval_iRandom'
                }
                self.set_methods_for_optimizer(methods_setting_dict)
        obj = self
        #for attr_name in dir(obj):
        #    if attr_name.startswith("__"):  # 組み込み属性（magic methodなど）をスキップ
        #        continue
        #    attr_value = getattr(obj, attr_name)
        #    #logging.debug(f"{attr_name}: {type(attr_value)}")
        #    print(f"{attr_name}: {type(attr_value)}")
        ###########シリアライズ判別だけ
        #for attr_name, attr_value in vars(obj).items():
        #    try:
        #        dill.dumps(attr_value)
        #    except Exception as e:
        #        print(f"Cannot pickle the attribute {attr_name}: {e}")
        init_spec_dict['target_obj'] = copy.deepcopy(self)
        self.optimizer = optimizer_class(hpara=hpara, init_spec_dict=init_spec_dict)
        self._optimized = False


    def _default_obs(self):
        '''
        とりあえずこれは使えない
        default observable for QNN (to be overwritten if necessary)
        '''
        obs = Observable(self.n_qubits)
        Z_string = ' '.join([f'Z {i}' for i in range(self.num_bit)])
        obs.add_operator(1., Z_string)
        #今回は部分トレース使わない（遅くなる疑惑）
        # if not hasattr(self, '_tr_out_ind'):
        #     self._tr_out_ind = [i for i in range(1, self.n_qubits)]        
        return obs      
            
    def _eval_state_exact_class_prob(self, c, state):
        '''
        とりあえずIris (3-class)専用
        exact expectation of the obs with the given state
        '''
        #caller = inspect.currentframe().f_back
        ##logging.debug(f"Called from function {caller.f_code.co_name} in {caller.f_code.co_filename}:{caller.f_lineno}")
        ##logging.debug('just before using self.observable')
        #obs = qulacs.observable.from_json(self.obs_json)
        #とりあえず3-classで
        obs = Observable(self.n_qubits)
        if self.class_unbias:
            #以下のPOVMでクラスのバイアスをなくす
            if c==0:
                obs.add_operator(1./3., "")  # Identity項 (1/2)
                obs.add_operator(1./3., "Z 0 Z 1")  # Z0とZ1の積 (Z0 Z1)
                obs.add_operator(1./6., "Z 0")
                obs.add_operator(1./6., "Z 1")
            if c==1:
                obs.add_operator(1./3., "")  # Identity項 (1/2)
                obs.add_operator(-1./3., "Z 0")  # Z0とZ1の積 (Z0 Z1)
                obs.add_operator(1./6., "Z 1")
                obs.add_operator(-1./6., "Z 0 Z 1")
            if c==2:
                obs.add_operator(1./3., "")  # Identity項 (1/2)
                obs.add_operator(-1./3., "Z 1")  # Z0とZ1の積 (Z0 Z1)
                obs.add_operator(1./6., "Z 0")
                obs.add_operator(-1./6., "Z 0 Z 1")
        else:
            if c==0:
                obs.add_operator(0.5, "")  # Identity項 (1/2)
                obs.add_operator(0.5, "Z 0 Z 1")  # Z0とZ1の積 (Z0 Z1)
            if c==1:
                obs.add_operator(0.25, '')
                obs.add_operator(-0.25, 'Z 0')
                obs.add_operator(0.25, 'Z 1')
                obs.add_operator(-0.25, 'Z 0 Z 1')
            if c==2:
                obs.add_operator(0.25, '')
                obs.add_operator(0.25, 'Z 0')
                obs.add_operator(-0.25, 'Z 1')
                obs.add_operator(-0.25, 'Z 0 Z 1')
        #tmp_state = state.copy()
        #traced_state = partial_trace(tmp_state, self._tr_out_ind)
        prob = obs.get_expectation_value(state)
        return (prob).real
    
    def exact_prob_dist_single_data(self, x):
        '''
        prob dist
        '''
        if self.noisy:
            if self.exact_clean:
                state = self.circuit.run(x, density=False, which=0) #0-th indexにcleanが入っている前提
            else:
                state = self.circuit.run(x, density=self.density)
        else:
            state = self.circuit.run(x)
        prob_dist = np.array([self._eval_state_exact_class_prob(c, state) for c in range(self.num_classes)])
        return prob_dist
    
    def exact_prob_dist_data_by_data(self, x_list, params):
        if self.noisy:
            if self.exact_clean:
                self.circuit.update_parameters_single(params, which=0) #0-th がclean前提
            else:
                self.circuit.update_parameters_single(params, which=None)
        else:
            self.circuit.update_parameters(params)
        prob_dist_list = np.array([self.exact_prob_dist_single_data(x) for x in x_list])
        return prob_dist_list
    
    def exact_predict_single_data(self, x, eps=0.):
        '''
        多数決の無限サンプルとして、最大確率をとるクラスとしてpredict
        option eps=0.: どれだけのエラーを許容することを要求するか。argmaxと次に大きいものが、2*eps以内の差しかなければ誤りとする。
        (確実に誤りとするために、-2を返す。-2はラベルとして使わないことが前提)
        つまり、surrogate作るときか、predictするときのサンプリングオーバーヘッドを1/eps^2 倍以下にしたいとき。
        '''
        if self.noisy:
            if self.exact_clean:
                state = self.circuit.run(x, density=False, which=0) #0-th indexにcleanが入っている前提
            else:
                state = self.circuit.run(x, density=self.density)
        else:
            state = self.circuit.run(x)
        prob_dist = [self._eval_state_exact_class_prob(c, state) for c in range(self.num_classes - 1)]
        prob_dist.append(1. - sum(prob_dist))
        i_max = np.argmax(prob_dist)
        p_2nd = np.max(np.delete(prob_dist, i_max))
        if prob_dist[i_max] - p_2nd < 2*eps:
            i_max = -1
        return i_max
    
    def exact_predict(self, x_list, params, eps=0.):
        if self.noisy:
            if self.exact_clean:
                self.circuit.update_parameters_single(params, which=0) #0-th がclean前提
            else:
                self.circuit.update_parameters_single(params, which=None)
        else:
            self.circuit.update_parameters(params)
        pred_list = np.array([self.exact_predict_single_data(x, eps=eps) for x in x_list])
        return pred_list
    
    def exact_error_rate(self, params, y_pred_dict=None, which="test", ret_both=False, log_evol=False, eps=0.):
        #logging.debug('exact_MSE just called')
        if ret_both:
            which="test"
        if which=="train":
            if y_pred_dict is None:
                y_pred = self.exact_predict(self.x_train, params, eps=eps).astype(float)
            else:
                y_pred = y_pred_dict['train']
            #logging.debug('just after exact_predict')
            y_train = self.y_train
            error = 1 - accuracy_score(y_train, y_pred)
            if log_evol:
                self.exact_cost_evol.append(self.exact_error_rate(params, which="test"))
            return error
        elif which=="test":
            #logging.debug('before test data processing')
            if y_pred_dict is None:
                y_pred = self.exact_predict(self.x_test, params, eps=eps).astype(float)
            else:
                y_pred = y_pred_dict['test']
            y_test = self.y_test
            error = 1 - accuracy_score(y_test, y_pred)
            if log_evol:
                self.exact_cost_evol.append(error)
            if ret_both:
                #logging.debug('just before call exact_MSE for train')
                error_train = self.exact_error_rate(params, which='train', ret_both=False)
                return {'test': error, 'train': error_train}
            else:
                return error
            
    def exact_error_both_to_track(self, params, y_pred_dict=None, eps=0.):
        '''
        error rate
        exactな trainとtest loss両方をtrack するとき用
        '''
        #logging.debug('MSE_both_exact just called')
        return self.exact_error_rate(params, y_pred_dict=y_pred_dict, which="test", ret_both=True, log_evol=False, eps=eps)
    
    def _y_pred_w_err(self, prob_dist_data_list, eps=0.):
        # 各確率分布の最大値と2番目に大きい値のインデックスを取得
        sorted_indices = np.argsort(prob_dist_data_list, axis=1)[:, -2:]
        top_values = np.take_along_axis(prob_dist_data_list, sorted_indices, axis=1)

        # 最大値と2番目に大きい値の差を計算
        diff = top_values[:, -1] - top_values[:, -2]

        # 条件に基づいて結果を計算
        result = np.where(diff > 2.*eps, sorted_indices[:, -1], -1)

        return result

    def exact_MSE(self, params, which="test", ret_both=False, log_evol=False, partial_MSE=False, ret_prediction=False, eps=0.):
        if ret_both:
            which="test"
        if which=="train":
            p_y_pred = self.exact_prob_dist_data_by_data(self.x_train, params).astype(float)
            if ret_prediction:
                y_pred = self._y_pred_w_err(p_y_pred, eps=eps)
            p_y_train = self.y_train_1hot
            if partial_MSE:
                p_y_pred = p_y_pred * p_y_train  # Zero out predictions for non-true classes
            MSE = np.sum(np.square(p_y_pred - p_y_train)) / len(p_y_train)
            if log_evol:
                self.exact_cost_evol.append(self.exact_MSE(params, which="test"))
            if ret_prediction:
                return MSE, y_pred
            else:
                return MSE
        elif which=="test":
            p_y_pred = self.exact_prob_dist_data_by_data(self.x_test, params).astype(float)
            if ret_prediction:
                y_pred_test = self._y_pred_w_err(p_y_pred, eps=eps)
            p_y_test = self.y_test_1hot
            if partial_MSE:
                p_y_pred = p_y_pred * p_y_test  # Zero out predictions for non-true classes
            MSE = np.sum(np.square(p_y_pred - p_y_test)) / len(p_y_test)
            if log_evol:
                self.exact_cost_evol.append(MSE)
            if ret_both:
                if ret_prediction:
                    MSE_train, y_pred_train = self.exact_MSE(params, which='train', ret_prediction=True, partial_MSE=partial_MSE, eps=eps)                    
                    y_pred_dict = {'train': y_pred_train, 'test': y_pred_test}
                    return [MSE, MSE_train], y_pred_dict
                else:
                    MSE_train = self.exact_MSE(params, which='train', ret_prediction=False)
                    return [MSE, MSE_train]
            else:
                if ret_prediction:
                    return MSE, y_pred_test
                else:
                    return MSE
    
    def exact_MAE(self, params, which="test", ret_both=False, log_evol=False, partial_MSE=False, ret_prediction=False):
        if ret_both:
            which="test"
        if which=="train":
            p_y_pred = self.exact_prob_dist_data_by_data(self.x_train, params).astype(float)
            if ret_prediction:
                y_pred = np.argmax(p_y_pred, axis=1)
            p_y_train = self.y_train_1hot
            if partial_MSE:
                p_y_pred = p_y_pred * p_y_train  # Zero out predictions for non-true classes
            MSE = np.sum(np.square(p_y_pred - p_y_train)) / len(p_y_train)
            if log_evol:
                self.exact_cost_evol.append(self.exact_MSE(params, which="test"))
            if ret_prediction:
                return MSE, y_pred
            else:
                return MSE
        elif which=="test":
            p_y_pred = self.exact_prob_dist_data_by_data(self.x_test, params).astype(float)
            if ret_prediction:
                y_pred_test = np.argmax(p_y_pred, axis=1)
            p_y_test = self.y_test_1hot
            if partial_MSE:
                p_y_pred = p_y_pred * p_y_test  # Zero out predictions for non-true classes
            MSE = np.sum(np.square(p_y_pred - p_y_test)) / len(p_y_test)
            if log_evol:
                self.exact_cost_evol.append(MSE)
            if ret_both:
                if ret_prediction:
                    MSE_train, y_pred_train = self.exact_MSE(params, which='train', ret_prediction=True)                    
                    y_pred_dict = {'train': y_pred_train, 'test': y_pred_test}
                    return [MSE, MSE_train], y_pred_dict
                else:
                    MSE_train = self.exact_MSE(params, which='train', ret_prediction=False)
                    return [MSE, MSE_train]
            else:
                if ret_prediction:
                    return MSE, y_pred_test
                else:
                    return MSE


    def exact_predict_error_and_MSE(self, params, loss="MSE", eps=None):
        '''        
        self.partial_MSE=Trueなら、正解ラベルとの距離だけを計算。
        lossを指定して、別のlossでも同じことをしようとしていた。: loss=
        '''
        if eps is None:
            eps = self._eps
        MSE_list, y_pred_dict = self.exact_MSE(params, which="test", ret_both=True, log_evol=False, partial_MSE=self.partial_MSE, ret_prediction=True, eps=eps)
        MSE_dict = {'test loss': MSE_list[0], 'train loss': MSE_list[1]}
        err_dict = self.exact_error_both_to_track(params, y_pred_dict=y_pred_dict) #{'test': error, 'train': error_train}        
        err_dict.update(MSE_dict)
        return err_dict    

    def _sample_state_Z(self, state, n_shots, Return_var=False, Return_outcomes=False):
        '''
        given state に対して、Zのサンプリングをする。基本的に期待値を返す。
        Zのqubit数を変えたい場合は書き換える。デフォルトでは1-qubitのZ0。
        Return_var=Trueなら、不偏分散の推定も返す。
        Return_outcomes=Trueなら、outcomesだけを直接返す。（Return_varは無効）
        '''
        self.call_counter[0] += 1
        #tmp_state = state.copy()
        #traced_state = partial_trace(tmp_state, self._tr_out_ind)
        samples = np.array(state.sampling(n_shots))
        #print(samples)
        outcomes = (-1) ** (samples % 2)
        #print(outcomes)
        self.shots_counter[0] += n_shots
        if Return_outcomes:
            return outcomes
        else:
            exp_val = np.mean(outcomes)
            if Return_var:
                var = float(n_shots) / (float(n_shots) - 1.) * (1. - exp_val**2) #サンプル不偏分散
                return exp_val, var
            else:
                return exp_val

    #データごとのobservableの期待値とべき乗のリストと分散    
    def _sample_prob_dist_data_by_data(self, params, x_eval, n_s_list, Return_var=True):
        '''
        データ列 x_eval (list)の各データすべてについて n_s_list の要素のショット数を使って、observableの測定をした期待値とべき乗を、
        スケーリングなしで返す。勾配計算のためにつかう。
        Return_var=Trueならその分散も返す
        期待値の配列exp_powers_listのshapeは、(order, len(x_eval))
        つまり、exp_powers_list[j]に、期待値のj+1乗の推定量の各データでの値のリストが入っている。
        分散の配列のshapeは、(len(x_eval),)
        
        '''
        self.circuit.update_parameters(params[:self.n_c_params])        
        if Return_var:
            p_list = np.zeros((len(x_eval), self.num_classes))
            p2_list = np.zeros((len(x_eval), self.num_classes))
            varp_list = np.zeros((len(x_eval), self.num_classes))
            for i in range(len(x_eval)):                
                p, p2, varp = self._sample_single_data_prob_dist(x_eval[i], n_s_list[i], Return_var=Return_var)            
                p_list[i] = p
                p2_list[i] = p2
                varp_list[i] = varp
            return p_list, p2_list, varp_list
        else:
            p_list = np.array([self._sample_single_data_prob_dist(x, n_s_list[i], Return_var=Return_var) for i, x in enumerate(x_eval)])
            return p_list
    
    def _sample_single_data_prob_dist(self, x, n_shots, Return_var=True):
        '''
        '''
        state = self.circuit.run(x)
        self.circuit_counter[0] += 1 #circuitをrunした場所
        return self._sample_state_prob_dist(state, n_shots, Return_var=Return_var)
    
    def _sample_state_prob_dist(self, state, n_shots, Return_var=True):
        '''
        '''
        self.call_counter[0] += 1
        #tmp_state = state.copy()
        #traced_state = partial_trace(tmp_state, self._tr_out_ind)
        samples = np.array(state.sampling(n_shots))
        if self.class_unbias:
            outcomes = samples % 4
            indices_of_3 = np.where(outcomes == 3)[0]
            outcomes[indices_of_3] = np.random.choice([0, 1, 2], size=len(indices_of_3))
        else:
            outcomes = (samples % 4) % 3
        self.shots_counter[0] += n_shots
        sample_bincount = np.bincount(outcomes, minlength=self.num_classes).astype(float)
        exp_prob_dist = sample_bincount / n_shots
        if Return_var:
            var_prob_list = sample_bincount / (n_shots - 1.) - np.square(sample_bincount) / (n_shots * (n_shots - 1.)) #各クラスのprob massのサンプル不偏分散
            n = n_shots
            p2 = (n/(n-1))*np.square(exp_prob_dist) - exp_prob_dist/(n-1)
            return exp_prob_dist, p2, var_prob_list
        else:
            return exp_prob_dist

    def _sample_state_class_prob(self, c, state, n_shots, reciprocal=False, Return_var=True, ret_outcomes=False):
        '''
        stateにおいて、class c の確率を、success prob推定する
        gradの推定ではcountを使って、ratioを直接推定する(probを計算しなくていい)
        分散の計算で、probやreciprocal、その分散を使うので、一緒に返す。
        '''
        self.call_counter[0] += 1
        #tmp_state = state.copy()
        #traced_state = partial_trace(tmp_state, self._tr_out_ind)
        samples = np.array(state.sampling(n_shots))
        if self.class_unbias:
            #unbiasにするには、POVMに対応して、3がでたら1/3の確率で他に分配
            outcomes = samples % 4
            indices_of_3 = np.where(outcomes == 3)[0]
            outcomes[indices_of_3] = np.random.choice([0, 1, 2], size=len(indices_of_3))
        else:
            outcomes = (samples % 4) % 3
        #print(samples)
        outcomes = (outcomes == c).astype(float)
        self.shots_counter[0] += n_shots
        if ret_outcomes:
            return outcomes
        #print(outcomes)        
        count = np.sum(outcomes)
        if not Return_var:
            return count
        if reciprocal:            
            theta = (n_shots + 1)/(count + 1)
            return count, theta
        else:
            n = n_shots
            p = np.mean(outcomes)
            varp = np.var(outcomes, ddof=1)
            p2 = (n/(n-1))*p**2 - p/(n-1)
            return count, p, p2, varp


    def _sample_single_data_class_prob(self, c, x, n_shots, reciprocal=False, Return_var=True, ret_outcomes=False,
                                        use_QEM=False, QEM_mathod='expZNE', shift=0, ind=0, use_sgm_12=False, QEM_unbiased_square=False):
        '''
        self.circuitにparameterがセットされた前提で、
        single dataで、observableのサンプル平均、サンプル不偏分散（Return_var=Trueのとき）を返す。
        powerつかわないやつ
        '''
        eps = 1e-8
        if use_QEM:
            state_list = self.circuit.run_multi(x, update_all=True) #state_listのindexとnoise_scalesのインデックスが一致しているように注意
            self.circuit_counter[0] += len(state_list)
            if QEM_mathod == 'expZNE':
                #2点のみ対応
                #noise_scale と、noiseの減衰パラメータの推定値と、それぞれでの分散から、各ノイズでの測定回数の分配を決める                
                mu = self.EMA_para
                if shift==0:
                    t = self.QEM_time_no_shift
                    noise_c = self.noise_c_no_shift
                    if use_sgm_12:
                        sgm_12 = self.sgm_12_no_shift
                    else:
                        sgm_12 = 1.
                else:
                    t = self.QEM_time[(shift+1)//2][ind]
                    noise_c = self.noise_c[(shift+1)//2][ind]
                    if use_sgm_12:
                        sgm_12 = self.sgm_12[(shift+1)//2][ind]
                    else:
                        sgm_12 = 1.
                if t > 0:
                    noise_c_u = noise_c/(1. - mu**t)
                    if use_sgm_12:
                        sgm_12 /= 1. - mu**t
                else:
                    noise_c_u = noise_c
                #sgm_rate1_2 = self.sigma_rate1_2
                lmd2 = self.noise_scales[1]
                N1 = np.clip(np.ceil(lmd2/(np.exp(noise_c_u*(1.-lmd2))/sgm_12 + lmd2)*n_shots).astype(int), self.n_ZNE_min, None)
                N2 = np.clip(np.ceil(1./(1. + sgm_12*lmd2*np.exp(-noise_c_u*(lmd2-1.)))*n_shots).astype(int), self.n_ZNE_min, None)
                #各ノイズでの測定値から、QEM推定値を計算、分散もそれぞれの分散から計算 self.noise_paraも更新
                if (not reciprocal):
                    if self.QEM_same_var:
                        outcomes1 = self._sample_state_class_prob(c, state_list[0], N1, reciprocal=reciprocal, Return_var=True, ret_outcomes=True)
                        outcomes2 = self._sample_state_class_prob(c, state_list[1], N2, reciprocal=reciprocal, Return_var=True, ret_outcomes=True)
                        centered_outcomes_all = np.concatenate([outcomes1, outcomes2])
                        y1 = np.mean(outcomes1)
                        y2 = np.mean(outcomes2)
                        count_dummy = 1
                        pp_dummy = 1
                    else:
                        _, y1, _, var_y1 = self._sample_state_class_prob(c, state_list[0], N1, reciprocal=reciprocal, Return_var=True, ret_outcomes=False)
                        count_dummy, y2, pp_dummy, var_y2 = self._sample_state_class_prob(c, state_list[1], N2, reciprocal=reciprocal, Return_var=True, ret_outcomes=False)
                    aa = self.noise_a_vect[c]
                    bb = np.sign(y1 - aa) * (np.abs(y1 - aa))**(lmd2/(lmd2-1.)) / (np.abs(y2 - aa + eps))**(1./(lmd2-1.)) #expZNEによる推定
                    p_est = aa + bb
                    if p_est <= 0 or p_est >= 1:
                        #unphysicalなら元の値を使う
                        p_est = y1
                    if use_sgm_12:
                        sgm_12_est = np.sqrt(var_y1/var_y2)
                    if np.abs(y1 - aa) > eps:
                        #もし y1 - aa が小さすぎたら noise_c の更新をスキップ
                        c_est = np.log(np.abs((y1 - aa)/(y2 - aa + eps)))/(lmd2 - 1.)                    
                        if t == 0:
                            noise_c = 0.
                            sgm_12 = 0.
                        t += 1
                        noise_c = mu * noise_c + (1. - mu) * c_est
                        noise_c_u = noise_c / (1.-mu**t)
                    if use_sgm_12:
                        sgm_12 = mu * sgm_12 + (1. - mu) * sgm_12_est
                    if shift==0:
                        self.QEM_time_no_shift = t                                            
                        self.noise_c_no_shift = noise_c
                        if use_sgm_12:
                            self.sgm_12_no_shift = sgm_12
                    else:
                        self.QEM_time[(shift+1)//2][ind] = t
                        self.noise_c[(shift+1)//2][ind] = noise_c
                        if use_sgm_12:
                            self.sgm_12[(shift+1)//2][ind] = sgm_12                    
                    if use_sgm_12:
                        var_est = ((np.sqrt(var_y1)*lmd2*np.exp(noise_c_u) + np.sqrt(var_y2)*np.exp(noise_c_u*lmd2))/(lmd2-1.))**2 #これはsgm_12使う場合
                    else:
                        if self.QEM_same_var:
                            self.ZNE_ovh = ((lmd2*np.exp(noise_c_u) + np.exp(noise_c_u*lmd2))/(lmd2-1.))**2
                            var_est = np.sum(np.square(centered_outcomes_all))/(len(centered_outcomes_all) - 1) #全部まとめて同じぐらいとして分散を推定。あとでショット数にオーバーヘッドかける
                        else:
                            var_est = (var_y1*lmd2*np.exp(noise_c_u) + var_y2*np.exp(noise_c_u*lmd2))*(lmd2*np.exp(noise_c_u) + np.exp(noise_c_u*lmd2))/(lmd2-1.)**2
                return count_dummy, p_est, pp_dummy, var_est
            elif QEM_mathod=='linZNE':
                #print('preprocess n_shots', n_shots)
                #print('ZNE_coeff_dist', self.ZNE_coeff_dist)
                n_shots_ZNE_list = np.clip(np.ceil(self.ZNE_coeff_dist*n_shots).astype(int), self.n_ZNE_min, None)
                #print('n_shots_ZNE_list', n_shots_ZNE_list)
                yi_list = []
                var_yi_list = []
                pp_list = []
                centered_outcomes_all = []
                for i, st in enumerate(state_list):
                    #全部のノイズスケールでのサンプル期待値（経験成功確率）と２乗と分散を計算
                    if self.QEM_same_var:
                        outcomes = self._sample_state_class_prob(c, st, n_shots_ZNE_list[i], reciprocal=reciprocal, Return_var=False, ret_outcomes=True)
                        yi = np.mean(outcomes)
                        centered_outcomes = outcomes - yi #平均０。これの分散が等しいと仮定する
                        centered_outcomes_all = np.concatenate([centered_outcomes_all, centered_outcomes])
                        yi_list.append(yi)
                        pp_list.append(yi**2) #とりあえず不偏推定はしない
                        count_dummy=1
                    else:
                        count_dummy, yi, pp_i, var_yi = self._sample_state_class_prob(c, st, n_shots_ZNE_list[i], reciprocal=reciprocal, Return_var=True, ret_outcomes=False)
                        yi_list.append(yi)
                        var_yi_list.append(var_yi)
                        pp_list.append(pp_i)
                p_est = np.dot(self.ZNE_coeff, yi_list)
                if p_est <= 0 or p_est >= 1:
                    #unphysicalなら元の値を使う
                    p_est = yi_list[0]
                if self.QEM_same_var:
                    var_est = np.sum(np.square(centered_outcomes_all))/(len(centered_outcomes_all) - 1) #全部まとめて同じぐらいとして分散を推定。あとでショット数にオーバーヘッドかける
                    #print(var_est)
                    #print(self.ZNE_coeff)
                else:
                    var_est = np.dot(self.ZNE_var_coeff, var_yi_list)
                if not QEM_unbiased_square:
                    pp_list = pp_list[0]
                return count_dummy, p_est, pp_list, var_est
        else:        
            state = self.circuit.run(x)
            self.circuit_counter[0] += 1 #circuitをrunした場所
            return self._sample_state_class_prob(c, state, n_shots, reciprocal=reciprocal, Return_var=Return_var, ret_outcomes=ret_outcomes)
                

    #データごとのobservableの期待値とべき乗のリストと分散    
    def _sample_success_prob_data_by_data(self, params, x_eval, n_s_list, y_eval, reciprocal=False, ret_outcomes=False, 
                                          use_QEM=False, QEM_mathod='linZNE', shift=0, ind=0, use_sgm_12=False):
        '''
        データ列 x_eval (list)の各データすべてについて n_s_list の要素のショット数を使って、observableの測定をした期待値とべき乗を、
        スケーリングなしで返す。勾配計算のためにつかう。
        Return_var=Trueならその分散も返す
        期待値の配列exp_powers_listのshapeは、(order, len(x_eval))
        つまり、exp_powers_list[j]に、期待値のj+1乗の推定量の各データでの値のリストが入っている。
        分散の配列のshapeは、(len(x_eval),)
        QEMの分散は、一般には分散の一次近似
        '''
        if use_QEM:
            self.circuit.update_parameters_multi(params[:self.n_c_params], update_all=True)
        else:
            self.circuit.update_parameters(params[:self.n_c_params])
        count_list = np.zeros(len(x_eval))
        if ret_outcomes:
            #outcomesとして、各データの正解クラスy_evalなら1、それ以外0が入ったリストをデータごとに入れたものを返す
            outcomes = [self._sample_single_data_class_prob(y_eval[i], x_eval[i], n_s_list[i], reciprocal=False, ret_outcomes=True) for i in range(len(x_eval))]
            outsum = np.array([np.sum(outs) for outs in outcomes])
            return outcomes, outsum
        if reciprocal:
            theta_list = np.zeros(len(x_eval))
            for i in range(len(x_eval)):
                count, theta = self._sample_single_data_class_prob(y_eval[i], x_eval[i], n_s_list[i], reciprocal, ret_outcomes=False)
                count_list[i] = count
                theta_list[i] = theta
            return count_list, theta_list
        else:            
            p_list = np.zeros(len(x_eval))
            p2_list = np.zeros(len(x_eval))
            varp_list = np.zeros(len(x_eval))
            for i in range(len(x_eval)):
                count, p, p2, varp = self._sample_single_data_class_prob(y_eval[i], x_eval[i], n_s_list[i], reciprocal, ret_outcomes=False, 
                                                                         use_QEM=use_QEM, QEM_mathod=QEM_mathod, shift=shift, ind=ind, use_sgm_12=use_sgm_12)
                #QEM使う場合は、pとvarpだけが有効
                count_list[i] = count
                p_list[i] = p
                p2_list[i] = p2
                varp_list[i] = varp
            return count_list, p_list, p2_list, varp_list
        
    def _sample_success_prob_no_var_data_by_data(self, params, x_eval, n_s_list, y_eval, reciprocal=False):
        '''
        データ列 x_eval (list)の各データすべてについて n_s_list の要素のショット数を使って、observableの測定をした期待値とべき乗を、
        スケーリングなしで返す。勾配計算のためにつかう。
        Return_var=Trueならその分散も返す
        期待値の配列exp_powers_listのshapeは、(order, len(x_eval))
        つまり、exp_powers_list[j]に、期待値のj+1乗の推定量の各データでの値のリストが入っている。
        分散の配列のshapeは、(len(x_eval),)
        
        '''
        self.circuit.update_parameters(params[:self.n_c_params])
        count_list = np.zeros(len(x_eval))
        for i in range(len(x_eval)):
            count = self._sample_single_data_class_prob(y_eval[i], x_eval[i], n_s_list[i], reciprocal, Return_var=False)
            count_list[i] = count
        return count_list
    
    def _estimate_MSEgrad_mean_condVar(self, pp, pp2, varpp, pm, pm2, varpm, p0, p02, varp0, y_eval_1hot=None, second_order=False, partial_MSE=True, 
                                       use_QEM=False, QEM_method='expZNE'):
        '''
        EVの計算. second order 未実装
        すべてデータごとの値の入ったアレイ
        '''
        if partial_MSE:
            if use_QEM:
                #QEM使うときは不偏推定は難しいので諦めて平均推定量の２乗使う
                vardp = varpp + varpm
                EV = np.mean((p0 - 1.)**2 * vardp + varp0 * (pp - pm)**2)
            else:
                vardp = varpp + varpm
                term1 = (p02 - 2*p0 + 1.) * vardp
                term2 = varp0 * (pp2 + pm2 - 2.*pp*pm)
                EV = np.mean(term1 + term2)
                if EV < 0:
                    EV = np.mean((p0 - 1.)**2 * vardp + varp0 * (pp - pm)**2)
        else:
            vardp = varpp + varpm
            term1 = (p02 + (-2*p0+1)*y_eval_1hot) * vardp
            term2 = varp0 * (pp2 + pm2 - 2.*pp*pm)
            EV = np.mean(np.sum(term1 + term2, axis=1))
            if EV < 0:
                EV = np.mean(np.sum((p0 - y_eval_1hot)**2 * vardp + varp0 * (pp - pm)**2, axis=1))
        return EV
    
    def MSE_iEval(self, params, n_shots_list, option_min_approx=None, option_weight=None, mini_batch_size=None, Max_n_shots0=True, second_order=True, partial_MSE=False,
                use_QEM=False, use_sgm_12=False, QEM_method='linZNE', QEM_same_var=True, QEM_only_no_shift=True):
        '''
        data_consistent_components=True　も入れてもいいが保留（各成分で違うデータをevalしていいかどうか）
        opt_renormは、re-normalization factorを最適化するかどうか
        n_shots_list は、勾配の各成分のestimationで、ミニバッチ内の全データに使うショット数。同じショット数を分配する。
        QEM_same_var: ノイズスケールによらず分散が同程度として、全部のデータを使って分散推定精度をあげる。（個別にやると精度がわるく分散が小さく計算されるとショット数が足りない）
        '''
#        logger = init_logging('iEval', raw_id_for_filename=True)
        self.partial_MSE= partial_MSE
        grad = np.zeros_like(params)
        gvar_list_b = np.zeros_like(params)
        gvar_list_EV = np.zeros_like(params)
        gvar_list_EVi = np.zeros_like(params)
        gvar_list_EV0 = np.zeros_like(params)
        self.QEM_same_var = QEM_same_var
        
        if option_min_approx is None:
            min_approx = self.min_approx_shots
        else:
            min_approx = option_min_approx
        if option_weight is None:
            wt_dist = self._uniform_wt.copy()
        else:
            wt_dist = option_weight.copy()
            
        size = mini_batch_size
        eval_ind = self._pop_mini_batch_ind(size)
        #train dataの重みのミニバッチ取り出し
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        #train dataからミニバッチを取り出す
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]
        y_eval_1hot = self.y_train_1hot[eval_ind]
        grad_data_list = np.zeros((len(params), len(x_eval)))

        ############まず、現在のパラメータ
        ############まず、現在のパラメータ        
        if use_QEM and self.QEM_time_no_shift ==0:
            if QEM_method=='expZNE':
                noise_c = self.noise_c_no_shift
                lmd2 = self.noise_scales[1]
                self.ZNE_ovh = ((lmd2*np.exp(noise_c) + np.exp(noise_c*lmd2))/(lmd2-1.))**2 #全部の分散同じとして推定する場合に使う
                #QEMの初回は全部の分散のオーバーヘッド倍(初期推定)を定数でするとして、ショット数を同じだけ定数倍する
                if not QEM_only_no_shift:
                    n_shots_list = np.ceil(np.array(n_shots_list) * ((lmd2*np.exp(noise_c) + np.exp(noise_c*lmd2))/(lmd2-1.))**2).astype(int)
            if QEM_method=='linZNE':
                lmd = self.noise_scales
                m = len(lmd)
                bar_l = np.mean(lmd)
                S_ll = np.sum(np.square(lmd-bar_l))
                self.ZNE_coeff = - bar_l * (lmd - bar_l)/S_ll + 1./m
                self.ZNE_coeff_dist = np.abs(self.ZNE_coeff)
                self.ZNE_var_coeff = self.ZNE_coeff_dist * np.sum(self.ZNE_coeff_dist) #最適な分配をするときの分散の係数。これで全体の分散(n_tot=1)/n_shotsとなる分散係数が求まる
                self.ZNE_coeff_dist /= np.sum(self.ZNE_coeff_dist) #ショット数分配に使う 
                self.ZNE_ovh = 1./m + bar_l**2 / S_ll #全部の分散同じとして推定する場合に使う
                if not QEM_only_no_shift:
                    n_shots_list = np.ceil(np.array(n_shots_list)*self.ZNE_ovh).astype(int) #オーバーヘッド倍
                #linZNEの場合は係数一定なので、最初だけ計算して属性に保持してしまう。また、最初はショット数をオーバーヘッド倍する。
                # オーバーヘッドは分散が違うと一定ではないので一応毎回考慮する(分配には使わない)
                self.QEM_time_no_shift = 1
        elif use_QEM and self.QEM_same_var and (not QEM_only_no_shift):
            n_shots_list = np.ceil(np.array(n_shots_list)*self.ZNE_ovh).astype(int) #オーバーヘッド倍
        n_shots0 = np.max(n_shots_list)
        if use_QEM and QEM_only_no_shift and self.QEM_same_var:
            n_shots0 *= self.ZNE_ovh
        n_shots_data_list0 = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        if partial_MSE:
            count0, p0, p02, varp0 = self._sample_success_prob_data_by_data(params, x_eval, n_shots_data_list0, y_eval, reciprocal=False, 
                                                                            use_QEM=use_QEM, shift=0, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
        else:
            p0, p02, varp0 = self._sample_prob_dist_data_by_data(params, x_eval, n_shots_data_list0, Return_var=True)
            err = p0 - y_eval_1hot
        #self.cache_train_pred = exp_pred0
        #self.cache_y_train_eval = y_eval
        #回路のパラメータの勾配
        if use_QEM and QEM_only_no_shift:
            use_QEM = False
        for i in range(len(params)):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            if partial_MSE:
                #正解ラベルの確率だけ返す
                countp, pp, pp2, varpp = self._sample_success_prob_data_by_data(params + ei, x_eval, n_shots_data_list, y_eval, reciprocal=False, 
                                                                                use_QEM=use_QEM, shift=1, ind=i, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
                countm, pm, pm2, varpm = self._sample_success_prob_data_by_data(params - ei, x_eval, n_shots_data_list, y_eval, reciprocal=False,
                                                                                 use_QEM=use_QEM, shift=-1, ind=i, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
                grad_data_list[i] = (p0 - 1.)*(pp - pm)
                grad[i] = np.mean(grad_data_list[i])
                gvar_list_EV[i] = self._estimate_MSEgrad_mean_condVar(pp, pp2, varpp, pm, pm2, varpm, p0, p02, varp0, partial_MSE=partial_MSE, 
                                                                      use_QEM=use_QEM, QEM_method=QEM_method)
            else:
                pp, pp2, varpp = self._sample_prob_dist_data_by_data(params + ei, x_eval, n_shots_data_list, Return_var=True)
                pm, pm2, varpm = self._sample_prob_dist_data_by_data(params - ei, x_eval, n_shots_data_list, Return_var=True)
                grad_data_list[i] = np.sum(err*(pp - pm), axis=1)
                grad[i] = np.mean(grad_data_list[i])
                #確率分布は別のeventごとに独立ではないので、修正が必要
                gvar_list_EV[i] = self._estimate_MSEgrad_mean_condVar(pp, pp2, varpp, pm, pm2, varpm, p0, p02, varp0, y_eval_1hot=y_eval_1hot, partial_MSE=partial_MSE)
        output = {
            'grad': grad, 
            'gvar_list_EV':gvar_list_EV,
            'gvar_list_EVi':gvar_list_EVi,
            'gvar_list_EV0':gvar_list_EV0,
            'gvar_list_b':gvar_list_b, 
            'grad_data_list':grad_data_list
        }
        return output

    # refoqusで使うために、全体の分散をサンプルから求めてしまうやつ。data点についてもランダムにサンプルを取っているときに使える    
    def _direct_sample_MSEgrad_varUB_whole(self, outcomes_whole_no_shift, outcomes_whole_p=None, outcomes_whole_m=None, partial_MSE=True, y_eval_1hot=None):
        '''
        partial_MSE=Trueのとき、
        outcomes_whole はそれぞれ、[i][j]要素に、i-th dataでの、j番目の測定のoutcomeが入っている。(各iにoutcomesのarrayが入っているとする。)
        このoutcomeは、正解ラベルなら1,それ以外0をとる
        0,p,mで、データ点ごとのショット数の分配は同じにしている前提
        partial_MSE=Falseは未実装
        '''
        if partial_MSE:
            outcomes_0 = np.concatenate(outcomes_whole_no_shift)
            outcomes_p = np.concatenate(outcomes_whole_p)
            outcomes_m = np.concatenate(outcomes_whole_m)
            grad_outcomes_data = (outcomes_0 - 1.) * (outcomes_p - outcomes_m)
        return np.var(grad_outcomes_data, ddof=1)
    
    def MSE_iEval_iRandom(self, params, n_shots_list, option_weight=None, partial_MSE=True):
        '''
        gradの各成分独立にdata点にショット数振り分けてサンプリングする。independent Random (iRandom) data sampling 
        (mini-batch ではないdata点の取り方として)
        先行研究の refoqus (gCANS) に使うため
        partial_MSE=Falseは未実装
        '''
        grad = np.zeros_like(params)
        gvar_list = np.zeros_like(params)

        if option_weight is None:
            wt_dist = self._uniform_wt_dist.copy()
        else:
            wt_dist = option_weight.copy()
            wt_dist = np.array(wt_dist) / np.sum(wt_dist) #確率分布にする。
        
        M_tot = self.num_train_data
        for i in range(len(params)):
            n_shots_data_list = np.random.multinomial(n_shots_list[i], wt_dist)
            x_eval = self.x_train[n_shots_data_list != 0]
            y_eval = self.y_train[n_shots_data_list != 0]
            n_shots_eval = n_shots_data_list[n_shots_data_list != 0]
            n = n_shots_list[i]
            #回路パラメータ
            outcomes0, outsum0 = self._sample_success_prob_data_by_data(params, x_eval, n_shots_eval, y_eval, reciprocal=False, ret_outcomes=True)
            #outsum0 = np.sum(outcomes0, axis=1)
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            outcomes_p, outsum_p = self._sample_success_prob_data_by_data(params + ei, x_eval, n_shots_eval, y_eval, reciprocal=False, ret_outcomes=True)
            outcomes_m, outsum_m = self._sample_success_prob_data_by_data(params - ei, x_eval, n_shots_eval, y_eval, reciprocal=False, ret_outcomes=True)
            #outsum_p = np.sum(outcomes_p, axis=1)
            #outsum_m = np.sum(outcomes_m, axis=1)
            grad[i] = np.sum((outsum0/(1. + (n-1)/M_tot) - 1.) * (outsum_p - outsum_m)/n) #積の取り扱い。各outcomesを、1/pして、和を取るときには重み倍だから、M_totがでる。
            gvar_list[i] = self._direct_sample_MSEgrad_varUB_whole(outcomes0,
                                                                    outcomes_whole_p=outcomes_p, outcomes_whole_m=outcomes_m)
        iEval_return = {'grad':grad, 'gvar_list':gvar_list}
        return iEval_return

    def MSE_grad_eval(self, params, n_shots_list, option_min_approx=None, option_weight=None, mini_batch_size=None, Max_n_shots0=True, second_order=True, partial_MSE=True,
                      use_QEM=False, use_sgm_12=False, QEM_method='expZNE', QEM_same_var=True, QEM_only_no_shift=True):
        '''
        data_consistent_components=True　も入れてもいいが保留（各成分で違うデータをevalしていいかどうか）
        opt_renormは、re-normalization factorを最適化するかどうか
        n_shots_list は、勾配の各成分のestimationで、ミニバッチ内の全データに使うショット数。同じショット数を分配する。
        '''
#        logger = init_logging('iEval', raw_id_for_filename=True)
        self.partial_MSE= partial_MSE
        grad = np.zeros_like(params)
        self.QEM_same_var = QEM_same_var
        
        if option_min_approx is None:
            min_approx = self.min_approx_shots
        else:
            min_approx = option_min_approx
        if option_weight is None:
            wt_dist = self._uniform_wt.copy()
        else:
            wt_dist = option_weight.copy()
        
        #print(mini_batch_size)
        size = mini_batch_size
        eval_ind = self._pop_mini_batch_ind(size)
        #train dataの重みのミニバッチ取り出し
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        #train dataからミニバッチを取り出す
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]
        y_eval_1hot = self.y_train_1hot[eval_ind]        

        ############まず、現在のパラメータ
        ############まず、現在のパラメータ
        if use_QEM and self.QEM_time_no_shift ==0:
            if QEM_method=='expZNE':
                noise_c = self.noise_c_no_shift
                lmd2 = self.noise_scales[1]
                #QEMの初回は全部の分散のオーバーヘッド倍(初期推定)を定数でするとして、ショット数を同じだけ定数倍する
                n_shots_list = np.ceil(np.array(n_shots_list) * ((lmd2*np.exp(noise_c) + np.exp(noise_c*lmd2))/(lmd2-1.))**2).astype(int)
            if QEM_method=='linZNE':
                lmd = self.noise_scales
                m = len(lmd)
                bar_l = np.mean(lmd)
                S_ll = np.sum(np.square(lmd-bar_l))
                self.ZNE_coeff = - bar_l * (lmd - bar_l)/S_ll + 1./m
                self.ZNE_coeff_dist = np.abs(self.ZNE_coeff)
                self.ZNE_var_coeff = self.ZNE_coeff_dist * np.sum(self.ZNE_coeff_dist) #最適な分配をするときの分散の係数。これで全体の分散(n_tot=1)/n_shotsとなる分散係数が求まる
                self.ZNE_coeff_dist /= np.sum(self.ZNE_coeff_dist) #ショット数分配に使う
                self.ZNE_ovh = (1./m + bar_l**2 / S_ll)
                if not QEM_only_no_shift:                
                    n_shots_list = np.ceil(np.array(n_shots_list)*self.ZNE_ovh).astype(int) #オーバーヘッド倍
                #linZNEの場合は係数一定なので、最初だけ計算して属性に保持してしまう。また、最初はショット数をオーバーヘッド倍する。
                # オーバーヘッドは分散が違うと一定ではないので一応毎回考慮する(分配には使わない)
                self.QEM_time_no_shift = 1
        elif use_QEM and self.QEM_same_var and (not QEM_only_no_shift):
            n_shots_list = np.ceil(np.array(n_shots_list)*self.ZNE_ovh).astype(int) #オーバーヘッド倍
        n_shots0 = np.max(n_shots_list) 
        if use_QEM and QEM_only_no_shift and self.QEM_same_var:
            n_shots0 *= self.ZNE_ovh
        n_shots_data_list0 = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        if partial_MSE:
            if use_QEM:
                count0, p0, p02, varp0 = self._sample_success_prob_data_by_data(params, x_eval, n_shots_data_list0, y_eval, reciprocal=False, 
                                                                            use_QEM=use_QEM, shift=0, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
            else:
                count0 = self._sample_success_prob_no_var_data_by_data(params, x_eval, n_shots_data_list0, y_eval, reciprocal=False)
                p0 = count0 / n_shots_data_list0
        else:
            p0 = self._sample_prob_dist_data_by_data(params, x_eval, n_shots_data_list0, Return_var=False)
            err = p0 - y_eval_1hot
        #self.cache_train_pred = exp_pred0
        #self.cache_y_train_eval = y_eval
        #回路のパラメータの勾配
        if use_QEM and QEM_only_no_shift:
            use_QEM = False
        for i in range(len(params)):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            if partial_MSE:
                if use_QEM:
                    _, pp, pp2, varpp = self._sample_success_prob_data_by_data(params + ei, x_eval, n_shots_data_list, y_eval, reciprocal=False, 
                                                                                use_QEM=use_QEM, shift=1, ind=i, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
                    _, pm, pm2, varpm = self._sample_success_prob_data_by_data(params - ei, x_eval, n_shots_data_list, y_eval, reciprocal=False,
                                                                                 use_QEM=use_QEM, shift=-1, ind=i, use_sgm_12=use_sgm_12, QEM_mathod=QEM_method)
                else:
                    countp = self._sample_success_prob_no_var_data_by_data(params + ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)
                    countm = self._sample_success_prob_no_var_data_by_data(params - ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)
                    pp = countp / n_shots_data_list
                    pm = countm / n_shots_data_list
                grad_data_list = (p0 - 1.)*(pp - pm)
                grad[i] = np.mean(grad_data_list)                
            else:
                pp = self._sample_prob_dist_data_by_data(params + ei, x_eval, n_shots_data_list, Return_var=False)
                pm = self._sample_prob_dist_data_by_data(params - ei, x_eval, n_shots_data_list, Return_var=False)
                grad_data_list = np.sum(err*(pp - pm), axis=1)
                grad[i] = np.mean(grad_data_list)
        return grad
                
    def _estimate_CEgrad_mean_condVar(self, theta, pp, pp2, varpp, pm, pm2, varpm, second_order=False):
        '''
        EVの計算. second order 未実装
        すべてデータごとの値の入ったアレイ
        '''
        term1 = theta**2 * (varpp + varpm)
        var_theta = theta**2 * (theta - 1.)
        term2 = var_theta * (pp2 + pm2 - 2.*pp*pm)
        EV = 0.25 * np.mean(term1 + term2)
        if EV < 0:
            EV = 0.25 * np.mean(term1 + var_theta * (pp - pm)**2)
        return EV

        
    def CE_iEval(self, params, n_shots_list, mini_batch_size=None, option_min_approx=None, option_weight=None, L2_reg_lmd=0.01, second_order=False, Max_n_shots0=True):
        grad = np.zeros_like(params)
        gvar_list_b = np.zeros_like(params)
        gvar_list_EV = np.zeros_like(params)
        gvar_list_EVi = np.zeros_like(params)
        gvar_list_EV0 = np.zeros_like(params)        

        if option_min_approx is None:
            min_approx = self.min_approx_shots
        else:
            min_approx = option_min_approx
        if option_weight is None:
            wt_dist = self._uniform_wt.copy()
        else:
            wt_dist = option_weight.copy()
        size = mini_batch_size
        #print(size)
        eval_ind = self._pop_mini_batch_ind(size)
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        #print(len(wt_dist))
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]
        
        grad_data_list = np.zeros((len(params), len(x_eval)))

        ############まず、現在のパラメータ
        n_shots0 = np.max(n_shots_list) 
        n_shots_data_list0 = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        count0, theta = self._sample_success_prob_data_by_data(params, x_eval, n_shots_data_list0, y_eval, reciprocal=True)
        #self.cache_train_pred = exp_pred0
        #self.cache_y_train_eval = y_eval
        #回路のパラメータの勾配
        for i in range(len(params)):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            countp, pp, pp2, varpp = self._sample_success_prob_data_by_data(params + ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)
            countm, pm, pm2, varpm = self._sample_success_prob_data_by_data(params - ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)

            grad_data_list = - 0.5 * (countp - countm)/(count0 + 1)*((n_shots_data_list0 + 1)/(n_shots_data_list + 1))
            grad[i] = np.mean(grad_data_list) + L2_reg_lmd * params[i] #/ len(grad_data_list)
            gvar_list_EV[i] = self._estimate_CEgrad_mean_condVar(theta, pp, pp2, varpp, pm, pm2, varpm)

        output = {
            'grad': grad, 
            'gvar_list_EV':gvar_list_EV,
            'gvar_list_EVi':gvar_list_EVi,
            'gvar_list_EV0':gvar_list_EV0,
            'gvar_list_b':gvar_list_b, 
            'grad_data_list':grad_data_list
        }
        return output

    def CE_grad_eval(self, params, n_shots_list, mini_batch_size=None, option_min_approx=None, option_weight=None, L2_reg_lmd=0.4, second_order=False, Max_n_shots0=True):
        grad = np.zeros_like(params)

        if option_min_approx is None:
            min_approx = self.min_approx_shots
        else:
            min_approx = option_min_approx
        if option_weight is None:
            wt_dist = self._uniform_wt.copy()
        else:
            wt_dist = option_weight.copy()
        size = mini_batch_size
        #print(size)
        eval_ind = self._pop_mini_batch_ind(size)
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        #print(len(wt_dist))
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]

        ############まず、現在のパラメータ（scale parameter の勾配）
        n_shots0 = np.max(n_shots_list) #scale parameter の勾配の測定は、現在のパラメータでの期待値だけでいい。そういうパラメータがある場合はこれでいい
        n_shots_data_list0 = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        count0 = self._sample_success_prob_no_var_data_by_data(params, x_eval, n_shots_data_list0, y_eval, reciprocal=True)
        #self.cache_train_pred = exp_pred0
        #self.cache_y_train_eval = y_eval
        #回路のパラメータの勾配
        for i in range(len(params)):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            countp = self._sample_success_prob_no_var_data_by_data(params + ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)
            countm = self._sample_success_prob_no_var_data_by_data(params - ei, x_eval, n_shots_data_list, y_eval, reciprocal=False)

            grad_data_list = - 0.5 * (countp - countm)/(count0 + 1)*((n_shots_data_list0 + 1)/(n_shots_data_list + 1))
            grad[i] = np.mean(grad_data_list) + L2_reg_lmd * params[i] #/ len(grad_data_list)

        return grad
        

    def CE_iEval_iRandom(self):
        pass

    def CE_loss_eval(self):
        pass
  
    def generate_labels(self, x_data, target_params):
        '''
        target_paramsが真のパラメータとなるように
        x_dataたちをQNNに与えたときのoutputたちをデータとして生成する。
        '''
        global generate_labels_flag
        generate_labels_flag = True
        return self.exact_predict(x_data, target_params) 

    @classmethod
    def grid_search_kCV(cls, optimizer_name, circuit, n_qubits, x_data, y_data, grid_para, fixed_para=None, loss='cross_entropy', options=None,
                        n_splits=5, parallel=False, processes=-1, select_min_score=None,
                        min_approx_shots=-1, n_params=None, init_params=None, Random_search=False, random_num=10, shuffle=False,
                        init_spec_dict=None, manual_methods_setting_dict=None, score_evaluator=None, random_seed_data = None, use_hpm=False,
                        shot_time=1.0, per_circuit_overhead=0.0, communication_overhead=0.0, return_all_results=True, partial_MSE=True, fit_options=None, **qnn_obj_kwargs):
        '''
        circuit : sk-qulacsのQNNとして、train paramsとembedding両方備わっているものを与える。
        grid_para : 辞書で、グリッドサーチする変数名をkeyとして、グリッドサーチする値のリストが入っている
                    例えば、{'alpha':[0.001, 0.01, 0.1], 'beta':[0.05, 0.02]} など
                    さらに、すべての組み合わせではなく連動させたい場合、例えば、Lの値に対し、alphaは、1/L, 1.5/L などとしたい場合、
                    grid_para = {'L': [0.1, 0.2, 0.3], 'alpha': ['1/L', '1.5/L']} のように、別のパラメータ名を変数名として含むコードを文字列として与えれば、そのような置き換えが実行できる。
        fixed_para : 固定のハイパーパラメータ
        init_params: Noneであれば、毎回ランダムに生成
        Random_search: Trueであれば、グリッドサーチ全部ではなく、ランダムにいくつかやる
        optimizer_name (str): VQA_optimizers に定義されている optimizer クラスの名前の文字列を指定。
        stratified k-fold cv で評価する: n_splits個にデータを分けて、一つをテストデータに使うことで、n_splits個のtrain-testの組ができるので、全部の汎化性能の平均をとることで評価する。
        評価のリストと、評価最大のパラメータを返す
        score_evaluator (callable or str): resultsを引数として、scoreとして、results_fval_listに入れる値を計算する関数またはそれを表す文字列。
            与えない場合は、'func_evol'の最後のテスト値を使う。
        kwargsは、最適化で使うgrad計算などのmethodに与えられる（どれも共通で与えられることに注意）
        (別々にキーワードを与えたい場合は工夫が必要。（未実装)）
        select_min_score デフォルトでNoneなのはlossから自動決定できるため

        Return: hps_results_dict keys are:
        if return_all_results:
            selected_params, selected_results, selected_fscore, hpm_results, hpm_fscore, results_fscore_list, param_list, selected_ind
        else:
            selected_params, selected_fscore, hpm_fscore, results_fscore_list, param_list, selected_ind
        hpmは、ハイパーパラメータと結果たちをまとめて保持し、ハイパーパラメータに対応する結果を引き出したり、ファイルからロードしたりするオブジェクト。
        '''
        #skf = StratifiedKFold(n_splits=n_splits) #contiだとだめ
        #train_test_inds = skf.split(x_data, y_data)
        start = time.time()
        if fit_options is None:
            fit_options = {}
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        if select_min_score is None:
            if loss == 'mean_squared_error':
                select_min_score = True
            elif loss == 'cross_entropy':
                select_min_score = True #error rateなどを用いる
            else:
                select_min_score = True
        if fixed_para is None:
            fixed_para = {}
        random_seed = random_seed_data
        if n_splits == 1:
            _, _, _, _, train_inds, test_inds = train_test_split(
                x_data, y_data, train_size=0.5, random_state=random_seed, return_indices=True
            )
            train_test_inds = [(train_inds, test_inds)]
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
            train_test_inds = kf.split(x_data)
        if n_params is None:
            n_params = len(circuit.get_parameters()) + 1
        #multiprocessingのときは必要
        if processes < 0:
            processes = cpu_count() - 1

        if score_evaluator is None:
            def latest_test_loss(results):
                return results['func_evol'][-1]['test']
            score_evaluator = latest_test_loss
        elif score_evaluator=='latest_test_loss_SA':
            options['take_SA'] = True
            def latest_test_loss_SA(results):
                return results['func_evol_SA'][-1]['test']
            score_evaluator = latest_test_loss_SA
        elif score_evaluator=='latest_test_loss_BMA':
            options['take_BMA'] = True
            def latest_test_loss_BMA(results):
                return results['func_evol_BMA'][-1]['test']
            score_evaluator = latest_test_loss_BMA

        idstr = str(uuid.uuid4())
        #単位処理
        logger = init_logging(f"{optimizer_name}_whole_progress_{idstr}", raw_id_for_filename=True, directory='whole_progress_log')
        #root_logger = getLogger()
        #for handler in root_logger.handlers:
        #    print(f"Handler: {handler}")
        #    print(f"Formatter: {handler.formatter}")
#         for handler in logger.handlers:
#             print(f"Handler: {handler}")
#             print(f"Formatter: {handler.formatter._fmt}")
#             print(f"Date format: {handler.formatter.datefmt}")
        def single_proc(args, idx):
            ##logging.basicConfig(level=#logging.DEBUG, filename='apps.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
            if init_params is None:
                init_para = np.random.uniform(-np.pi, np.pi, n_params)
                #init_para[-1] = 1.
            else:
                init_para = init_params
            x_train = x_data[train_inds] #train data として使うデータたち（複数）のインデックスたちのリストが指定される
            y_train = y_data[train_inds]
            x_test = x_data[test_inds]
            y_test = y_data[test_inds]
            file_id = str(args['para_set_id']) + '_' + str(args['dataset_id']) + '_' + idstr
            ##logging.debug('just before construct qnn')
            qnn = cls(n_qubits, circuit, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, file_id=file_id,
                            min_approx_shots=min_approx_shots, 
                            shot_time=shot_time, per_circuit_overhead=per_circuit_overhead, communication_overhead=communication_overhead,
                            **qnn_obj_kwargs)
            hpara = fixed_para.copy()
            hpara.update(args['search_para'])
            #logging.debug('just before compile')
            qnn.compile(optimizer_name, loss, hpara=hpara, init_params=init_para,
                        init_spec_dict=init_spec_dict, manual_methods_setting_dict=manual_methods_setting_dict, partial_MSE=partial_MSE)
            results = qnn.fit(options=options, auto_save=False, process_id=args, **fit_options)
            if return_all_results:
                return results, idx
            else:
                return score_evaluator(results), idx #最後のテストスコアの値を返す
        #１度にわたす引数のリストをつくる。grid searchの一つの組につき、k-fold cv やる回数分だけデータセットのわけかたについてバリエーションがある
        param_list = []
        for para_comb in itertools.product(*grid_para.values()):
            param_dict = dict(zip(grid_para.keys(), para_comb))
            # 数式を含むパラメータを評価
            for key, value in param_dict.items():
                if isinstance(value, str):
                    if '***uncode' in value:
                        param_dict[key] = value.replace('***uncode', '')
                    else:
                        param_dict[key] = evaluate_expression(value, param_dict)
            param_list.append(param_dict)
        if Random_search:
            param_list = list(np.random.choice(param_list, random_num, replace=False))
        
        print('total number of grid points:', len(param_list))
        param_sets = []
        for j, (train_inds, test_inds) in enumerate(train_test_inds):
            for i, pst in enumerate(param_list):
                pset = {'search_para':pst.copy(), 'train_inds':train_inds, 'test_inds':test_inds, 'para_set_id':i, 'dataset_id':j}
                param_sets.append(pset)

        # 途中で止まった場合などに備えて、一回ずつの最適化結果をファイルから読み出すのに必要な param_setsを保存しておく。
        # params_setsを読み込んで、para_set_idやdataset_idを参照すれば、ファイルがどれに対応するかわかる。idstrがさらに個別の回を特定する。
        directory_p_sets = 'HPS_hparam_sets'
        file_name = f'param_sets_{idstr}'
        if directory_p_sets:
            # Make sure the directory exists. If not, create it.
            if not os.path.exists(directory_p_sets):
                os.makedirs(directory_p_sets)
            file_name = f"{directory_p_sets}/{file_name}.pkl"
        else:
            file_name = file_name
        with open(file_name, 'wb') as f:
            dill.dump(param_sets, f)
        print(f"param_sets saved to file: {file_name}")
        #client = Client()
        #args_bag = db.from_sequence(param_sets)
        #results = args_bag.map(single_proc).compute()
        #joblibでもdillができないからだめ。PyCapsuleができない-> dillを自分で定義(バージョン？のせいでだめっぽい)
        #clouddill
        #serialized_function = dill.dumps(single_proc)
        #results_list = joblib.Parallel(n_jobs=processes, backend='dill ')([joblib.delayed(single_proc)(args) for args in param_sets])
        #multiprocessingを使うと、quantum circuitのクラスがdillできないといわれる
        total_process_num = len(param_sets)
        process_count = 0
        results_list = [None] * total_process_num
        best_score = None
        start_time = time.time()
        if parallel:
            print('Use with OpenMP may cause a deadlock; be careful of the value of OMP_NUM_THREADS! (OMP_NUM_THREADS=1 is recommended.)')
            with ProcessingPool(nodes=processes) as p:
                for result, idx in p.uimap(single_proc, param_sets, range(len(param_sets))):
                    results_list[idx] = result
                    process_count += 1
                    logger.info(f'{process_count} processes out of {total_process_num} completed ({100*process_count/total_process_num} %).')
                    elapsed_time = time.time() - start_time
                    speed = process_count/elapsed_time
                    remain_num = total_process_num - process_count
                    estimated_time = remain_num / speed
                    remain_rounds = np.ceil(remain_num / processes)
                    p_estimated_time = remain_rounds / speed
                    logger.info(f"Speed: {speed:.4f} tasks/sec, Simply estimated time remaining: {seconds2hms(estimated_time)}")
                    logger.info(f"Parallel estimated time remaining: {seconds2hms(p_estimated_time)}")
                    tmp_score = result['func_evol'][-1]['test']
                    hash_id = hashlib.md5(str(param_sets[idx]).encode()).hexdigest()
                    para_set_id = param_sets[idx]['para_set_id']
                    if best_score is None:
                        best_score = tmp_score                        
                        logger.info(f'pset: {param_sets[idx]} \n \n fileID: {hash_id} \n para_index: {para_set_id} \n Best score: {best_score} \n')
                    elif best_score > tmp_score:
                        best_score = tmp_score
                        #para_set_id = param_sets[idx]['para_set_id']
                        logger.info(f'Best score updated: pset: {param_sets[idx]} \n \n fileID: {hash_id} \n para_index: {para_set_id} \n Best score: {best_score} \n')
                    else:
                        logger.info(f'fileID: {hash_id} \n para_set_id: {para_set_id} \n score: {tmp_score} \n')
        else:
            for result, idx in map(single_proc, param_sets, range(len(param_sets))):
                results_list[idx] = result
                process_count += 1
                logger.info(f'{process_count} processes out of {total_process_num} completed ({100*process_count/total_process_num} %).')
                tmp_score = result['func_evol'][-1]['test']
                hash_id = hashlib.md5(str(param_sets[idx]).encode()).hexdigest()
                if best_score is None:
                    best_score = tmp_score
                    logger.info(f'Best score: {best_score} \n pset: {param_sets[idx]} \n \n fileID: {hash_id}')
                elif best_score > tmp_score:
                    best_score = tmp_score
                    logger.info(f'Updated Best score: {best_score} \n pset: {param_sets[idx]} \n \n fileID: {hash_id}')
                else:
                    logger.info(f'fileID: {hash_id} \n score: {tmp_score} \n')
        
        results_dict = {}
        for result, param_set in zip(results_list, param_sets):
            #param_setsの１要素param_setは、ハイパーパラメータの組と、train, test の分け方一つで１要素だが、同じハイパーパラメータを表すidがpara_set_idに入っている。
            para_set_id = param_set['para_set_id']
            if para_set_id not in results_dict:
                results_dict[para_set_id] = []
            results_dict[para_set_id].append(result) #ハイパーパラメータが同じで、データの分け方が違うものは、同じキーにリストとしてまとめて入る。
        if return_all_results:
            results_fval_dict = {}
            for result, param_set in zip(results_list, param_sets):
                para_set_id = param_set['para_set_id']
                #print(para_set_id)
                if para_set_id not in results_fval_dict:
                    results_fval_dict[para_set_id] = []
                results_fval_dict[para_set_id].append(score_evaluator(result))

        if not use_hpm:
            hpm_fscore = 'no_hpm'
            hpm_results = 'no_hpm'
        if return_all_results:
            if use_hpm:
                hpm_results = HyperParameterManager()
                hpm_fscore = HyperParameterManager(base_dir='hp_fscore', base_dir_metadata='hp_fscore_metadata')
            results_list = []
            results_fscore_list = []
            for i in range(len(param_list)):
                results_list.append(results_dict[i]) #インデックスが、param_listと連動したリストになっている
                fscore = np.mean(results_fval_dict[i])
                if use_hpm:
                    hpm_results.add_result(param_list[i], results_dict[i]) #CVのそれぞれのresultsたちのリストが一つのhparaセットの項目として加わる
                    hpm_fscore.add_result(param_list[i], fscore)
                results_fscore_list.append(fscore)
        else:
            if use_hpm:
                hpm_fscore = HyperParameterManager(base_dir='hp_fscore', base_dir_metadata='hp_fscore_metadata')
            results_fscore_list = []
            for i in range(len(param_list)):
                fscore = np.mean(results_fval_dict[i])
                if use_hpm:
                    hpm_fscore.add_result(param_list[i], fscore)
                results_fscore_list.append(fscore)
        if select_min_score:
            selected_ind = np.argmin(results_fscore_list)
        else:
            selected_ind = np.argmax(results_fscore_list)
        selected_params = param_list[selected_ind]
        selected_fscore = results_fscore_list[selected_ind]
        elapsed_time = time.time() - start
        print('selected hpara:', selected_params)
        print('selected score:', selected_fscore)
        print('all fscore list:', results_fscore_list)
        print('elapsed_time:', elapsed_time)
        logger.info(f'selected hpara: {selected_params}')
        logger.info(f'selected score: {selected_fscore}')
        logger.info(f'all fscore list: {results_fscore_list}')
        logger.info(f'elapsed_time: {elapsed_time}')
        hps_results_dict = {
            'selected_params': selected_params,
            'selected_fscore': selected_fscore,
            'hpm_fscore': hpm_fscore,
            'results_fscore_list': results_fscore_list,
            'param_list': param_list,
            'selected_ind': selected_ind,
            'elapsed_time': elapsed_time
        }

        if return_all_results:
            selected_results = results_list[selected_ind]
            hps_results_dict.update({
                'selected_results': selected_results,
                'hpm_results': hpm_results,
                'results_list': results_list
            })

        return hps_results_dict
    
    @classmethod
    def compare_optimizers_kCV(cls, optimizers, circuit, n_qubits, x_data, y_data, loss='mean_squared_error', n_splits=5, repeat_kCV=True,
                        num_trials_per_combination=1, init_params=None, hpara_dict=None, optimization_options=None, compile_options=None, fit_kwargs_dict=None,
                        parallel=False, processes=-1, min_approx_shots=-1, n_params=None, random_seed_data=None, shuffle=False,                      
                        manual_methods_setting_dict=None, save_file_name=None, save_directory=None, optimizer_str_map=None,
                        shot_time=1.0, per_circuit_overhead=0.0, communication_overhead=0.0, **qnn_obj_kwargs):
        '''
        各種optionsは、optimizerに固有の値を使いたいオプションはoptimizer名のキーで、共通のオプションは直下に記述できる。
        例えば、{'SGD': {'option1': val1}, 'option1': common_val1, 'option2': common_val2} のように記述できる。
        compile_optionsに共通の'loss'を与えていたとしても、loss引数が優先される。
        hpara_dict: optimizerをキーとして、そのoptimizerのハイパーパラメータたちの辞書を与える。

        SAやBMAをとりたいときは、optimization_optionsでtake_SA=True, take_BMA=Trueを指定。そのときのハイパーパラメータは、hpara_dictで指定。
        '''
        start = time.time()
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        random_seed = random_seed_data
        num_trials_per_combination_cache = num_trials_per_combination
        if compile_options is None:
            compile_options = {}
        compile_options['loss'] = loss
        if n_splits == 1:
            _, _, _, _, train_inds, test_inds = train_test_split(
                x_data, y_data, train_size=0.7, random_state=random_seed, return_indices=True
            )
            train_test_inds = [(train_inds, test_inds)]
        else:
            if repeat_kCV:
                train_test_inds = []
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
                for i in range(num_trials_per_combination):                    
                    train_test_inds.extend(kf.split(x_data))
                num_trials_per_combination = 1
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
                train_test_inds = kf.split(x_data)
        if n_params is None:
            n_params = len(circuit.get_parameters()) + 1
        #multiprocessingのときは必要
        if processes < 0:
            processes = cpu_count() - 1
        base_qnn = cls(n_qubits, circuit, min_approx_shots=min_approx_shots, 
                                    setting_dict=manual_methods_setting_dict, 
                                    shot_time=shot_time, per_circuit_overhead=per_circuit_overhead, communication_overhead=communication_overhead,
                                    **qnn_obj_kwargs)
        opt_trials = OptimizationTrialsManager(base_qnn, x_data, y_data)
        #各試行のデータを準備する。
        opt_trials.bulk_add_trials(optimizers, train_test_inds, num_trials_per_combination, init_params, hpara_dict, optimization_options, compile_options, fit_kwargs_dict, optimizer_str_map=optimizer_str_map)
        idstr = str(uuid.uuid4())
        logger = init_logging(f"compare_whole_progress_{idstr}", raw_id_for_filename=True, directory='whole_progress_log')
        total_process_num = n_splits*num_trials_per_combination_cache*len(optimizers)
        process_count = 0
        best_score = None
        start_time = time.time()
        trial_names = list(opt_trials.trials.keys())
        if parallel:
            print('Use with OpenMP may cause a deadlock; be careful of the value of OMP_NUM_THREADS! (OMP_NUM_THREADS=1 is recommended.)')            
            with ProcessingPool(nodes=processes) as p:
                for (trial_name, results, qnn_instance, init_params) in p.uimap(opt_trials.optimize, trial_names):
                    opt_trials.trials[trial_name].update({
                        "results": results,
                        "qnn_instance": qnn_instance,
                        "init_params": init_params
                    })
                    process_count += 1
                    logger.info(f'{process_count} processes out of {total_process_num} completed ({100*process_count/total_process_num} %).')
                    elapsed_time = time.time() - start_time
                    speed = process_count/elapsed_time
                    remain_num = total_process_num - process_count
                    estimated_time = remain_num / speed
                    remain_rounds = np.ceil(remain_num / processes)
                    #p_estimated_time = remain_rounds / speed
                    logger.info(f"Speed: {speed:.4f} tasks/sec, Simply estimated time remaining: {seconds2hms(estimated_time)}")
                    logger.info(f"remain rounds: {remain_rounds}")
                    tmp_score = results['func_evol'][-1]['test']
                    #hash_id = hashlib.md5(str(param_sets[idx]).encode()).hexdigest()
                    if best_score is None:
                        best_score = tmp_score                        
                        logger.info(f'trial name: {trial_name} \n Best score: {best_score} \n')
                    elif best_score > tmp_score:
                        best_score = tmp_score                        
                        logger.info(f'Best score updated! \n \n trial_name {trial_name} \n Best score: {best_score} \n')
                    else:
                        logger.info(f'trial name: {trial_name} \n score: {tmp_score} \n')
        else:
            for (trial_name, results, qnn_instance, init_params) in map(opt_trials.optimize, trial_names):
                    opt_trials.trials[trial_name].update({
                        "results": results,
                        "qnn_instance": qnn_instance,
                        "init_params": init_params
                    })
                    process_count += 1
                    logger.info(f'{process_count} processes out of {total_process_num} completed ({100*process_count/total_process_num} %).')
                    elapsed_time = time.time() - start_time
                    speed = process_count/elapsed_time
                    remain_num = total_process_num - process_count
                    estimated_time = remain_num / speed
                    remain_rounds = np.ceil(remain_num / processes)
                    #p_estimated_time = remain_rounds / speed
                    logger.info(f"Speed: {speed:.4f} tasks/sec, Simply estimated time remaining: {seconds2hms(estimated_time)}")
                    logger.info(f"remain rounds: {remain_rounds}")
                    tmp_score = results['func_evol'][-1]['test']
                    #hash_id = hashlib.md5(str(param_sets[idx]).encode()).hexdigest()
                    if best_score is None:
                        best_score = tmp_score                        
                        logger.info(f'trial name: {trial_name} \n Best score: {best_score} \n')
                    elif best_score > tmp_score:
                        best_score = tmp_score                        
                        logger.info(f'Best score updated! \n \n trial_name {trial_name} \n Best score: {best_score} \n')
                    else:
                        logger.info(f'trial name: {trial_name} \n score: {tmp_score} \n')            
        opt_trials.save(filename=save_file_name, directory=save_directory)
        elapsed_time = time.time() - start
        print(elapsed_time)
        return opt_trials, elapsed_time
    
def continue_optimization_trials(opt_trials, new_optimization_options, fit_kwargs_dict=None, parallel=False, processes=-1):
    start = time.time()
    opt_trials.update_trials_options_and_kwargs(optimization_options=new_optimization_options, kwargs_dict=fit_kwargs_dict)
    if processes < 0:
            processes = cpu_count() - 1
    if parallel:
        print('Use with OpenMP may cause a deadlock; be careful of the value of OMP_NUM_THREADS! (OMP_NUM_THREADS=1 is recommended.)')
        trial_names = list(opt_trials.trials.keys())
        with ProcessingPool(nodes=processes) as p:
            for (trial_name, results, qnn_instance, init_params) in p.uimap(opt_trials.optimize, trial_names):
                opt_trials.trials[trial_name].update({
                    "results": results,
                    "qnn_instance": qnn_instance,
                    "init_params": init_params
                })
    else:
        list(map(opt_trials.optimize, opt_trials.trials.keys()))
    elapsed_time = time.time() - start
    print(elapsed_time)
    return opt_trials, elapsed_time

def seconds2hms(seconds):
    days = int(seconds // 86400)
    time_struct = time.gmtime(seconds)
    return f"{days} d: {time_struct.tm_hour} h: {time_struct.tm_min} m: {time_struct.tm_sec} s:"

class OptimizationTrialsManager:
    def __init__(self, base_qnn_instance=None, base_x_data=None, base_y_data=None, default_directory="optimization_trial_data", default_filename=None, load_from_file=False, load_filepath=None):
        '''
        データとcircuitがある程度大きくても複数回の試行を保存して管理するためのクラス
        最適化の試行の内部状態を保持、保存しつつ、追加データもとれるようにする
        さらにビッグデータの場合は、圧縮したりファイルを媒介するなど工夫が必要
        各trialのデータを self.trials[trial_name]に保持
        circuitは、self._base_qnn_instanceに保持し、各trialのqnn_instanceには持たせない
        データは、base_x_data, base_y_dataを与えた場合、self._base_x_data, self._base_y_data に保存し、各trialにはindexだけを保持。
            その場合、_base_qnn のtrain, testデータは削除される
        base_x_data, base_y_dataを与えない場合、自動的に _base_qnn から読み込んだtrain, test データを使う
        （base dataを、_base_qnnから構成しようかとも思ったが、そうするとindexにずれが生じる危険性があるので、index指定方式を使いたいときは、explicitにbaseを与えることを強制する。）
        使い方の注意としては、base_x_data, base_y_data を与えた場合は、base_qnn のデータは削除されるので、必ずtrialごとにindexを与えないといけないこと。
        trialごとにqnn_instanceを与えることはできない。
        fit の auto_save のデータを探すときには、ファイルの後ろが、trial_name + base_qnnのfile_idとなっているものを探す。
        '''
        self.trials = {}
        self.metadata = {}
        if load_from_file:
            if load_filepath:
                self.load(filename=load_filepath, directory=None)  # directory=None で完全なパスを指定
                return
            else:
                raise ValueError("load_filepath must be provided when load_from_file is True.")
        else:
            if base_qnn_instance:
                self._base_qnn = base_qnn_instance
            else:
                raise ValueError("base_qnn_instance must be provided when not loading from file.")
            if (not (base_x_data is None)) and (not (base_y_data is None)):
                self._base_x_data = np.array(base_x_data)
                self._base_y_data = np.array(base_y_data)
                self._base_qnn.remove_data()
                print('Explicit base data are given. Note that the data are removed from the base QNN object.\n',
                      'You MUST give indices of train/test data to each trial')
            else:
                self._base_x_data = None
                self._base_y_data = None        
        
        self.default_directory = default_directory
        if default_filename is None:
            unique_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            self.default_filename = f"trials_{timestamp}_{unique_id}.pkl"
        else:
            self.default_filename = default_filename

    def add_trial(self, trial_name, optimizer_str, train_inds=None, test_inds=None, init_params=None, hpara=None, optimization_options=None, compile_options=None, **kwargs):
        '''
        **kwargsは、fitに与えられる
        '''
        if compile_options is None:
            compile_options = {}
        trial_qnn = copy.deepcopy(self._base_qnn)
        if (not (train_inds is None)) or (not (test_inds is None)):
            if self._base_x_data is None or self._base_y_data is None:
                print('Warning: Base data are not given. train_inds and test_inds are ignored.')
        if train_inds is not None and test_inds is not None and self._base_x_data is not None and self._base_y_data is not None:
            x_train = self._base_x_data[train_inds] #train data として使うデータたち（複数）のインデックスたちのリストが指定される
            y_train = self._base_y_data[train_inds]
            x_test = self._base_x_data[test_inds]
            y_test = self._base_y_data[test_inds]
            trial_qnn.set_data(x_train, y_train, x_test, y_test)
        elif not all(trial_qnn.get_data()):
            print('Error: No data are set.')
            return None
        trial_qnn.file_id = trial_name + trial_qnn.file_id #1回ごとの試行のバックアップファイルの識別用(fitのauto_save)
        filtered_compile_options = filter_by_signature(trial_qnn.compile, compile_options, exclude_keys=['optimizer'])
        trial_qnn.compile(optimizer=optimizer_str, hpara=hpara, init_params=init_params, **filtered_compile_options)
        
        self.trials[trial_name] = {
            'qnn_instance': trial_qnn,
            'train_inds': train_inds,
            'test_inds': test_inds,
            'init_params': init_params,
            'options': optimization_options,
            'hpara': hpara,
            'compile_options': compile_options,
            'kwargs': kwargs,
        }

        # リソースを節約するため、circuitは保存しない
        self.trials[trial_name]['qnn_instance'].circuit = None
        self.trials[trial_name]['qnn_instance'].remove_data()

    def optimize(self, trial_name):
        '''
        trialごとに別のインスタンスを使うことになるから、並列処理する場合は、これを単位処理としてそのまま使えば良い。
        '''
        trial = self.trials[trial_name]
        qnn_instance = trial['qnn_instance']
        
        # circuitを再設定
        qnn_instance.circuit = self._base_qnn.circuit
        train_inds = trial['train_inds']
        test_inds = trial['test_inds']
        if  train_inds is not None and test_inds is not None and self._base_x_data is not None and self._base_y_data is not None:
            x_train = self._base_x_data[train_inds] #train data として使うデータたち（複数）のインデックスたちのリストが指定される
            y_train = self._base_y_data[train_inds]
            x_test = self._base_x_data[test_inds]
            y_test = self._base_y_data[test_inds]
            qnn_instance.set_data(x_train, y_train, x_test, y_test)
        else:
            qnn_instance.set_data(*self._base_qnn.get_data())
        
        results = qnn_instance.fit(
            init_params=trial['init_params'],
            options=trial['options'],
            hpara=trial['hpara'],
            process_id = trial_name,
            **trial['kwargs']
        )
        trial['results'] = results
        init_params = qnn_instance.optimizer.init_params
        trial['init_params'] = init_params
        
        # リソースを節約するため、データ削除して保持する
        qnn_instance.circuit = None
        qnn_instance.remove_data()
        trial['qnn_instance'] = qnn_instance
        #並列処理のときのために、値を返す。並列処理をすると、独立なコピーができて変更が反映されない恐れがある。
        # (multiprocessingとかだとまさにそうなる)
        return trial_name, results, qnn_instance, init_params

    def save(self, filename=None, directory=None):
        save_dir = directory if directory else self.default_directory
        
        # ディレクトリが存在しない場合、作成する
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        optimizers = '_'.join(self.metadata.keys())
        save_filename = filename if filename else (optimizers + '_' + self.default_filename)
        save_path = os.path.join(save_dir, save_filename)

        with open(save_path, 'wb') as file:
            try:
                dill.dump((self._base_qnn, self._base_x_data, self._base_y_data, self.trials, self.metadata), file)
            except Exception as e:
                dill.dump(('dummy', self._base_x_data, self._base_y_data, self.trials, self.metadata), file)

    def load(self, filename=None, directory=None):
        load_dir = directory if directory else ''
        load_filename = filename if filename else self.default_filename
        load_path = os.path.join(load_dir, load_filename)

        with open(load_path, 'rb') as file:
            self._base_qnn, self._base_x_data, self._base_y_data, loaded_trials, self.metadata = dill.load(file)
            
            for trial_name, trial_data in loaded_trials.items():
                self.trials[trial_name] = trial_data

    def bulk_add_trials(self, optimizers, data_splits=None, num_trials_per_combination=1, init_params=None, hpara_dict=None, optimization_options=None, compile_options=None, kwargs_dict=None, optimizer_str_map=None):
        '''
        optimizers: List of optimizers as strings.
        data_splits: Optional list of tuples, where each tuple has two arrays/lists, (train_inds, test_inds). Defaults to None.
        num_trials_per_combination: Number of trials to run for each optimizer-data_split combination.
        init_params は、[trial_num][split_id][optimizer] の任意の階層まで指定して共通のパラメータを指定できる。
            例えば、単にinit_paramsそのものを指定したら全て共通。
            init_params[trial_num][split_id] = params のように指定したら、optimizerたちで共通の初期値のtrialsを実行。(これが実用上多い)
            init_params[trial_num][optimizer] = params のように指定することも可能。
            ただし、[trial_num][split_id][optimizer]　の順番の階層でなければならないことに注意。
            デフォルトでは、[trial_num] ごとにランダムに初期化する。
        各種optionsは、optimizerに固有の値を使いたいオプションはoptimizer名のキーで、共通のオプションは直下に記述できる。
        例えば、{'SGD': {'option1': val1}, 'option1': common_val1, 'option2': common_val2} のように記述できる
        Returns a metadata dictionary which maps the optimizer and data split ID to the generated trial names.
        '''
        if optimizer_str_map is None:
            optimizer_str_map = {}

        if not data_splits:
            data_splits = [(None, None)] # Use the default data in the _base_qnn
        if not hpara_dict:
            hpara_dict = {}
        if not optimization_options:
            optimization_options = {}
        if not compile_options:
            compile_options = {}
        if not kwargs_dict:
            kwargs_dict = {}
        n_params = self._base_qnn.get_parameter_count()
        init_params_cache = {}

        local_metadata = {}
        for optimizer in optimizers:
            local_metadata[optimizer] = {}
            hpara = hpara_dict.get(optimizer, None)
            # optimization_options の更新
            specific_optimization_options = optimization_options.get(optimizer, {})
            common_optimization_options = {k: v for k, v in optimization_options.items() if k not in optimizers}
            optimizer_optimization_options = {**common_optimization_options, **specific_optimization_options}

            # compile_options の更新
            specific_compile_options = compile_options.get(optimizer, {})
            common_compile_options = {k: v for k, v in compile_options.items() if k not in optimizers}
            optimizer_compile_options = {**common_compile_options, **specific_compile_options}

            # kwargs_dict の更新
            specific_kwargs = kwargs_dict.get(optimizer, {})
            common_kwargs = {k: v for k, v in kwargs_dict.items() if k not in optimizers}
            optimizer_kwargs = {**common_kwargs, **specific_kwargs}
            
            for split_id, (train_inds, test_inds) in enumerate(data_splits):
                local_metadata[optimizer][split_id] = []

                for trial_num in range(num_trials_per_combination):
                    trial_name = f"{optimizer}_split{split_id}_trial{trial_num}"

                    # init_paramsの処理
                    if init_params is None:
                        if (trial_num, split_id) not in init_params_cache:
                            init_para = np.random.uniform(-np.pi, np.pi, n_params)
                            #init_para[-1] = 1.
                            init_params_cache[(trial_num, split_id)] = init_para
                        params = init_params_cache[(trial_num, split_id)]
                    else:
                        depth = get_depth(init_params)
                        if depth == 1:
                            params = init_params
                        elif depth == 2:
                            if isinstance(init_params, (list, np.ndarray)):
                                params = init_params[trial_num]
                            elif optimizer in init_params:
                                params = init_params[optimizer]
                            else:
                                if (trial_num, split_id) not in init_params_cache:
                                    init_params_cache[(trial_num, split_id)] = np.random.uniform(-np.pi, np.pi, n_params)
                                params = init_params_cache[(trial_num, split_id)]
                        elif depth == 3:
                            if isinstance(init_params[trial_num], (list, np.ndarray)):
                                params = init_params[trial_num][split_id]
                            elif optimizer in init_params[trial_num]:
                                params = init_params[trial_num][optimizer]
                            else:
                                if (trial_num, split_id) not in init_params_cache:
                                    init_params_cache[(trial_num, split_id)] = np.random.uniform(-np.pi, np.pi, n_params)
                                params = init_params_cache[(trial_num, split_id)]
                        elif depth == 4 and optimizer in init_params[trial_num][split_id]:
                            params = init_params[trial_num][split_id][optimizer]
                        else:
                            if (trial_num, split_id) not in init_params_cache:
                                init_params_cache[(trial_num, split_id)] = np.random.uniform(-np.pi, np.pi, n_params)
                            params = init_params_cache[(trial_num, split_id)]
                    if optimizer in optimizer_str_map:
                        optimizer_str = optimizer_str_map[optimizer]
                    else:
                        optimizer_str = optimizer
                    self.add_trial(
                        trial_name=trial_name,
                        optimizer_str=optimizer_str,
                        train_inds=train_inds,
                        test_inds=test_inds,
                        init_params=params,
                        hpara=hpara,
                        optimization_options=optimizer_optimization_options,
                        compile_options=optimizer_compile_options,
                        **optimizer_kwargs
                    )
                    local_metadata[optimizer][split_id].append(trial_name)

        self.metadata = local_metadata
        return local_metadata
    
    def update_trials_options_and_kwargs(self, optimization_options=None, kwargs_dict=None):
        '''
        Update options and kwargs for all trials using optimization_options and kwargs_dict.

        optimization_options and kwargs_dict can be specified in the same way as in bulk_add_trials.
        This function uses the metadata to determine the corresponding optimizer and split_id for each trial.
        '''
        if not hasattr(self, 'metadata') or not self.metadata:
            print('No metadata. Nothing done.')
            return
        
        if not optimization_options:
            optimization_options = {}
            
        if not kwargs_dict:
            kwargs_dict = {}

        # Iterate over each optimizer in metadata
        for optimizer, split_data in self.metadata.items():

            # optimization_options update
            specific_optimization_options = optimization_options.get(optimizer, {})
            common_optimization_options = {k: v for k, v in optimization_options.items() if k not in self.metadata}
            optimizer_optimization_options = {**common_optimization_options, **specific_optimization_options}

            # kwargs_dict update
            specific_kwargs = kwargs_dict.get(optimizer, {})
            common_kwargs = {k: v for k, v in kwargs_dict.items() if k not in self.metadata}
            optimizer_kwargs = {**common_kwargs, **specific_kwargs}

            # Iterate over each split_id in split_data
            for split_id_list in split_data:
                for trial_name in split_id_list:

                    trial = self.trials[trial_name]

                    # Update the options and kwargs for the trial
                    trial['options'] = optimizer_optimization_options
                    trial['kwargs'] = optimizer_kwargs


def get_depth(obj):
    if isinstance(obj, (list, np.ndarray)) and len(obj) > 0:
        return 1 + get_depth(obj[0])
    elif isinstance(obj, dict) and len(obj) > 0:
        key = next(iter(obj))
        return 1 + get_depth(obj[key])
    else:
        return 0
    
def filter_by_signature(func, original_dict, exclude_keys=[]):
    # シグネチャから引数のリストを取得
    valid_keys = list(inspect.signature(func).parameters.keys())
    
    # 除外キーリストに基づいてvalid_keysからキーを除外
    for key in exclude_keys:
        if key in valid_keys:
            valid_keys.remove(key)
    
    # valid_keys に基づいて辞書をフィルタリング
    return {k: v for k, v in original_dict.items() if k in valid_keys}


import matplotlib.pyplot as plt

class OptimizationResultsAnalyzer:
    def __init__(self, opt_trials=None, default_file_dir=None):
        '''
        opt_trials: instance of OptimizationTrialsManager who has the results of optimizations to analyze
        '''
        self.cv_means_for_trials = {}
        self.cv_medians_for_trials = {}
        self.cv_mean_of_means = {}
        self.cv_median_of_means = {}
        self.cv_mean_of_medians = {}
        self.cv_median_of_medians = {}
        if not default_file_dir:
            self.default_dir = 'opt_fig'
        self.file_id = str(uuid.uuid4())[:8]
        if opt_trials is None:
            return
        if isinstance(opt_trials, list):
            instances = opt_trials
        else:
            instances = [opt_trials]
        merged_metadata = {}
        merged_trials = {}
        for instance in instances:
            merged_metadata.update(instance.metadata)
            merged_trials.update(instance.trials)
        self.metadata = merged_metadata
        self.trials = merged_trials

        # 2. 従来のoptimizerの取得部分を修正
        self.optimizers = list(merged_metadata.keys())
        

    def load_data_from_results_dict(self, input_data):
        '''
        主にgrid searchをした結果の一つのhparaの結果たちを直接読み込ませてプロットする用
        resultsのリストをoptimizerごとにまとめたdictを読み込ませて、OptimizerTrialsManagerインスタンスをつくって読み込ませる。
        input_data ={'SGD': [SGDresults1, SGDresults2,..], 'Adam': [Adamresults1, Adamresults2,...]}のような辞書。
        または、複数のresults_dictを入れたリスト。
        '''

        if isinstance(input_data, dict):
            # 単一のdictの場合
            results_dicts = [input_data]
        elif isinstance(input_data, list):
            # 複数のresults_dictを入れたリストの場合
            results_dicts = input_data
        else:
            raise ValueError("Invalid type for input_data. Expected a dict or a list of dicts.")

        instances = []
        for results_dict in results_dicts:
            opt_trials = OptimizationTrialsManager(base_qnn_instance='dummy')
            local_metadata = {}
            for optimizer, results_list in results_dict.items():
                local_metadata[optimizer] = {}
                for split_id, results in enumerate(results_list):
                    trial_name = f"{optimizer}_split{split_id}_trial{0}"
                    local_metadata[optimizer][split_id] = [trial_name]
                    if not trial_name in opt_trials.trials:
                        opt_trials.trials[trial_name] = {}
                    opt_trials.trials[trial_name]['results'] = results
            opt_trials.metadata = local_metadata
            instances.append(opt_trials)

        merged_metadata = {}
        merged_trials = {}
        for instance in instances:
            merged_metadata.update(instance.metadata)
            merged_trials.update(instance.trials)
        self.optimizers = list(merged_metadata.keys())
        self.metadata = merged_metadata
        self.trials = merged_trials


    def cv_all_trials(self, x_label='total_shots', y_labels=None, stat_proc="mean_and_median", end=None):
        '''
        各optimizerにおける全部のcvとtrialsに対する統計をとる(trialごとに別のsplitをするのが通常のrepeated cv)
        mean/std and/or median/iqr のdfを、それぞれ self.cv_mean[optimizer]['stat']/['spread'], self.cv_median[optimizer]['stat']/['spread']に入れる
        '''
        if y_labels is None:
            y_labels = [('func_evol', ''), ('func_evol_SA', 'SA'), ('func_evol_BMA', 'BMA')]
        if isinstance(y_labels, str):
            y_labels = [(y_labels, '')]

        if stat_proc not in ["mean", "median", "mean_and_median"]:
            raise ValueError("Invalid stat_proc option. Choose from 'mean', 'median', or 'mean_and_median'.")

        self.cv_mean = {}
        self.cv_median = {}

        for optimizer in self.optimizers:
            trial_names = [trial for sublist in self.metadata[optimizer].values() for trial in sublist]
            results_list = [self.trials[trial_name]['results'] for trial_name in trial_names]
            
            for y_label, y_suffix in y_labels:
                if y_suffix:
                    key_suffix = "_" + y_suffix
                else:
                    key_suffix = ""

                if stat_proc in ["mean", "mean_and_median"]:
                    mean_df, spread_df_mean = self.stat_proc_data(results_list, x_label, y_label, "mean", end=end)
                    if optimizer + key_suffix not in self.cv_mean:
                        self.cv_mean[optimizer + key_suffix] = {}
                    self.cv_mean[optimizer + key_suffix] = {"stat": mean_df, "spread": spread_df_mean}

                if stat_proc in ["median", "mean_and_median"]:
                    median_df, spread_df_median = self.stat_proc_data(results_list, x_label, y_label, "median", end=end)
                    if optimizer + key_suffix not in self.cv_median:
                        self.cv_median[optimizer + key_suffix] = {}
                    self.cv_median[optimizer + key_suffix] = {"stat": median_df, "spread": spread_df_median}

    def cv_per_trial(self, x_label='total_shots', y_labels=None, stat_proc="mean_and_median"):
        if y_labels is None:
            y_labels = [('func_evol', ''), ('func_evol_SA', 'SA'), ('func_evol_BMA', 'BMA')]
        if isinstance(y_labels, str):
            y_labels = [(y_labels, '')]

        if stat_proc not in ["mean", "median", "mean_and_median"]:
            raise ValueError("Invalid stat_proc option. Choose from 'mean', 'median', or 'mean_and_median'.")

        self.cv_means_for_trials = {} if stat_proc in ["mean", "mean_and_median"] else None
        self.cv_medians_for_trials = {} if stat_proc in ["median", "mean_and_median"] else None
        for optimizer in self.metadata.keys():
            trial_nums = len(self.metadata[optimizer][0])
            for trial_num in range(trial_nums):
                trial_names = [single_split_trials[trial_num] for single_split_trials in self.metadata[optimizer]]
                results_list = [self.trials[trial_name]['results'] for trial_name in trial_names]
                for y_label, y_suffix in y_labels:
                    if y_suffix:
                        key_suffix = "_" + y_suffix
                    else:
                        key_suffix = ""

                    if not (self.cv_means_for_trials is None):
                        mean_df, spread_df_mean = self.stat_proc_data(results_list, x_label, y_label, "mean")
                        if optimizer + key_suffix not in self.cv_means_for_trials:
                            self.cv_means_for_trials[optimizer + key_suffix] = {}
                        self.cv_means_for_trials[optimizer + key_suffix][trial_num] = {"stat": mean_df, "spread": spread_df_mean}

                    if not (self.cv_medians_for_trials is None):
                        median_df, spread_df_median = self.stat_proc_data(results_list, x_label, y_label, "median")
                        if optimizer + key_suffix not in self.cv_medians_for_trials:
                            self.cv_medians_for_trials[optimizer + key_suffix] = {}
                        self.cv_medians_for_trials[optimizer + key_suffix][trial_num] = {"stat": median_df, "spread": spread_df_median}
        if self.cv_means_for_trials:
            for key in self.cv_means_for_trials:
                print(key)
                self.cv_mean_of_means[key], self.cv_median_of_means[key] = compute_stats(self.cv_means_for_trials[key])

        if self.cv_medians_for_trials:
            for key in self.cv_medians_for_trials:
                print(key)
                self.cv_mean_of_medians[key], self.cv_median_of_medians[key] = compute_stats(self.cv_medians_for_trials[key])

    def _extract_evol(self, data_list):
        # データが辞書型かどうかを判定
        if data_list and isinstance(data_list[0], dict):
            return pd.DataFrame(data_list)
        else:
            return pd.DataFrame({'value': data_list})

    def stat_proc_data(self, results_list, x_label='total_shots', y_label='func_evol', stat_proc="median", unit=1, end=None):
        df_list = []
        for single_result in results_list:
            result_df = self._extract_evol(single_result[y_label])
            result_df.index = np.array(single_result[x_label]) * unit
            if end is not None:
                result_df = result_df.loc[0:end]
            df_list.append(result_df)
        
        df = pd.concat(df_list, axis='columns', keys=[f'data_{i}' for i in range(len(df_list))], sort=True)
        df = df.fillna(method='ffill')
        
        # カラム階層が2の場合と1の場合で平均取得の方法を変える
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) == 2:
            if stat_proc == "median":
                median_df = df.median(axis=1, level=1)
                #print(df.groupby(level=1, axis=1))
                #dfgroup = df.groupby(axis=1, level=1, sort=False)
                grouped_df = df.groupby(axis=1, level=1, sort=False)
                upper = grouped_df.apply(lambda group: group.quantile(0.75, axis=1))
                lower = grouped_df.apply(lambda group: group.quantile(0.25, axis=1))
                return median_df, (lower, upper)
            elif stat_proc == "mean":
                mean_df = df.mean(axis=1, level=1)
                std_df = df.std(axis=1, level=1)
                return mean_df, std_df
            else:
                return df, None
        else:
            if stat_proc == "median":
                median_df = df.median(axis=1)
                upper = df.quantile(0.75, axis=1)
                lower = df.quantile(0.25, axis=1)
                return median_df, (lower, upper)
            elif stat_proc == "mean":
                mean_df = df.mean(axis=1)
                std_df = df.std(axis=1)
                return mean_df, std_df
            else:
                return df, None
            
    def _get_optimizer_by_trial_name(self, trial_name):
        for optimizer, trials in self.metadata.items():
            for trial_group in trials.values():
                if trial_name in trial_group:
                    return optimizer
        return None
    
    def _get_split_id_by_trial_name(self, trial_name):
        for trials_opt in self.metadata.values():
            for split_id, trials_single_split in enumerate(trials_opt):
                if trial_name in trials_single_split:
                    return split_id
        return None
    
    def get_file_path_to_savefig(self, directory=None, fig_spec_str='', extension='svg'):
        if not directory:
            directory = self.default_dir
        return file_path_factory(directory=directory, given_name=None, file_name_str_list=[fig_spec_str, self.file_id], extension=extension)

    def plot_all_trials(self, x_label='total_shots', y_labels=None, x_label_name=None, y_label_name='', graph_title='', include_title=False,
                        single_split_data=False, split_id=0, save_figure=False, directory=None, file_extension='svg', xlim=None, ylim=None, xlog=False, ylog=False):
        '''
        y_labels can be a single label string or a list of tuple such as
        [('func_evol', ''), ('func_evol_SA', 'SA'), ('func_evol_BMA', 'BMA')]
        The second element of each tuple is used for the legend like
        "SGD_SA_train"
        single_split_data (bool): If it is True, this method plots only the trials with the given split_id
        ylim (tuple): (ymin, ymax) of the graph
        '''        
        if y_labels is None:
            y_labels = [('func_evol', ''), ('func_evol_SA', 'SA'), ('func_evol_BMA', 'BMA')]
        if isinstance(y_labels, str):
            y_labels = [(y_labels, '')]
        optimizers = self.optimizers
        # 1. tab10からoptimizerの基本色を取得
        base_colors = sns.color_palette("tab10", len(optimizers))
        optimizer_color_map = {optimizer: color for optimizer, color in zip(optimizers, base_colors)}

        line_styles = ['-', '--', '-.', ':']
        

        for data_type, ax_title in [("train", "Training Data"), ("test", "Testing Data")]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            created_labels = set()

            for trial_name, trial_data in self.trials.items():
                if single_split_data:
                    if self._get_split_id_by_trial_name(trial_name) != split_id:
                        continue
                optimizer = self._get_optimizer_by_trial_name(trial_name)
                results = trial_data['results']

                for idx, (y_label, y_legend_suffix) in enumerate(y_labels):
                    if y_label not in results:
                        continue
                    # 2. 各optimizerの異なるsuffixに対して色の変動を追加
                    # 色の変動を少し強めるための調整パラメータ
                    adjustment = 0.05
                    # optimizerの基本色を取得
                    base_h, base_l, base_s = optimizer_color_map[optimizer]
                    # HLS色空間で色の明度を変更
                    new_h = min(1,base_h + adjustment*idx)
                    new_l = max(0, base_l - adjustment * (idx + 1))
                    new_s = min(1,base_s + adjustment*idx)
                    color = (new_h, new_l, new_s)
                    
                    line_style = line_styles[idx % len(line_styles)]

                    data = [entry[data_type] if isinstance(entry, dict) else entry for entry in results[y_label]]
                    label = f"{optimizer} {y_legend_suffix} ({data_type})"

                    if label not in created_labels:
                        ax.plot(results[x_label], data, linestyle=line_style, color=color, label=label)
                        created_labels.add(label)
                    else:
                        ax.plot(results[x_label], data, linestyle=line_style, color=color)

            # 軸やラベルの設定
            ax.set_xlabel(x_label_name, fontsize=18)
            ax.set_ylabel(y_label_name, fontsize=18)
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
            if xlog:
                    ax.set_xscale('log')
            if ylog:
                ax.set_yscale('log')
            if include_title:
                ax.set_title(ax_title, fontsize=18)
            else:
                print(f"title: {ax_title}")
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.rc('legend', fontsize=18)
            ax.legend()

            ax.xaxis.get_offset_text().set(size=18)
            ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 6))

            plt.tight_layout()
            if save_figure:
                if single_split_data:
                    split_name = '_' + str(split_id)
                else:
                    split_name = ''
                file_path = self.get_file_path_to_savefig(directory=directory, fig_spec_str=f'all_trials_{data_type}{split_name}', extension=file_extension)
                plt.savefig(file_path)  # trainとtestの画像をそれぞれ保存
            plt.show()

    def plot_statistics(self, data_to_plot=['cv_mean', 'cv_median'], x_label_name=None, y_label_name='', graph_title='', include_title=False,
                        save_figure=False, directory=None, file_extension='svg', xlim=None, ylim=None, xlog=False, ylog=False, label_mapping=None, yticks=None,
                        line_style_mapping=None, no_filling_list=None):
        '''
        include_title=Trueのときだけ、グラフにタイトルを含める。Falseのときはprintする。（実際にプレゼンなどに使うときはタイトルはいらない事が多いため）
        '''
        optimizers = self.optimizers
        base_colors = sns.color_palette("tab10", len(optimizers))
        optimizer_color_map = {optimizer: color for optimizer, color in zip(optimizers, base_colors)}

        line_styles = ['-', '--', '-.', ':']
        data_list = [
            ('cv_mean', self.cv_mean), ('cv_median', self.cv_median),
            ("cv_means_for_trials", self.cv_means_for_trials), ("cv_medians_for_trials", self.cv_medians_for_trials),
            ("cv_mean_of_means", self.cv_mean_of_means), ("cv_median_of_means", self.cv_median_of_means),
            ("cv_mean_of_medians", self.cv_mean_of_medians), ("cv_median_of_medians", self.cv_median_of_medians)
        ]
        suffix_count = 1
        suffix_dict = {'': 0}

        for data_name, data_dict in data_list:
            if data_name not in data_to_plot:
                continue

            # Get columns from the first available dictionary for loop iteration
            sample_dict = list(data_dict.values())[0]
            if "stat" in sample_dict:
                sample_stat = sample_dict["stat"]
            else:
                sample_stat = list(sample_dict.values())[0]["stat"]
            columns_to_plot = sample_stat.columns

            for column in columns_to_plot:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.rc('legend',fontsize=18)
                created_labels = set()
                for optimizer_w_suff, value in data_dict.items():
                    if optimizer_w_suff not in optimizers:
                        optimizer_suffix = optimizer_w_suff.split("_")[-1] #splitで、_でわけた文字列のリストになる。
                        optimizer = "_".join(optimizer_w_suff.split("_")[:-1]) #_で文字列のリストを結合するという意味。
                    else:
                        optimizer_suffix = ""
                        optimizer = optimizer_w_suff
                    if not optimizer_suffix in suffix_dict.keys():
                        suffix_dict[optimizer_suffix] = suffix_count
                        suffix_count += 1
                    idx = suffix_dict[optimizer_suffix]
                    # 2. 各optimizerの異なるsuffixに対して色の変動を追加
                    # 色の変動を少し強めるための調整パラメータ
                    adjustment = 0.05
                    # optimizerの基本色を取得
                    base_h, base_l, base_s = optimizer_color_map[optimizer]
                    # HLS色空間で色の明度を変更
                    new_h = min(1,base_h + adjustment*idx)
                    new_l = max(0, base_l - adjustment * (idx + 1))
                    new_s = min(1,base_s + adjustment*idx)
                    color = (new_h, new_l, new_s)

                    line_style = line_styles[idx % len(line_styles)]
                    if label_mapping and optimizer in label_mapping:
                        optimizer_w_suff = label_mapping[optimizer]
                    if line_style_mapping and optimizer in line_style_mapping:
                        line_style = line_style_mapping[optimizer]
                    label=f"{optimizer_w_suff} ({column})"
                    
                    if data_name in ["cv_means_for_trials", "cv_medians_for_trials"]:
                        for trial_num, trial_data in value.items():
                            mean_data = trial_data["stat"][column]
                            if data_name == "cv_medians_for_trials":
                                lower_bound = trial_data["spread"][0][column]
                                upper_bound = trial_data["spread"][1][column]
                            else:
                                spread_data = trial_data["spread"][column]
                                lower_bound = mean_data - spread_data
                                upper_bound = mean_data + spread_data

                            if label not in created_labels:
                                ax.plot(mean_data, label=label, color=color, linestyle=line_style)
                                if not (no_filling_list and optimizer in no_filling_list):
                                   ax.fill_between(mean_data.index, lower_bound, upper_bound, color=color, alpha=0.2)
                                created_labels.add(label)
                            else:
                                ax.plot(mean_data, color=color, linestyle=line_style)
                                if not (no_filling_list and optimizer in no_filling_list):
                                    ax.fill_between(mean_data.index, lower_bound, upper_bound, color=color, alpha=0.2)
                    else:
                        mean_data = value["stat"][column]
                        if data_name == "cv_median":
                            lower_bound = value["spread"][0][column]
                            upper_bound = value["spread"][1][column]
                        else:
                            spread_data = value["spread"][column]
                            lower_bound = mean_data - spread_data
                            upper_bound = mean_data + spread_data                        
                        #spread_data = value["spread"][column]
                        ax.plot(mean_data, label=label, color=color, linestyle=line_style)
                        if not (no_filling_list and optimizer in no_filling_list):
                            ax.fill_between(mean_data.index, lower_bound, upper_bound, color=color, alpha=0.2)

                ax.xaxis.get_offset_text().set(size=18)
                ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 6))
                if yticks is not None:
                    ax.set_yticks(yticks)
                if xlim is not None:
                    ax.set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
                if not x_label_name:
                    ax.set_xlabel(x_label_name, fontsize=18)
                if not y_label_name:
                    ax.set_ylabel(y_label_name, fontsize=18)
                if include_title:
                    ax.set_title(f"{graph_title} ({data_name}) ({column})", fontsize=18)
                else:
                    print(f"title: {graph_title} ({data_name}) ({column})")                
                if xlog:
                    ax.set_xscale('log')
                if ylog:
                    ax.set_yscale('log')
                ax.legend()
                plt.tight_layout()
                if save_figure:
                    file_path = self.get_file_path_to_savefig(directory=directory, fig_spec_str=f'plt_stat_{data_name}_{column}', extension=file_extension)
                    plt.savefig(file_path)  # trainとtestの画像をそれぞれ保存
                plt.show()                         
            

def compute_stats(df_dict_optimizer):
    concatenated_stats = pd.concat([df_dict['stat'] for df_dict in df_dict_optimizer.values()], axis='columns', keys=range(len(df_dict_optimizer)), sort=True)
    concatenated_stats = concatenated_stats.fillna(method='ffill')
    #print(concatenated_stats)
    is_multiindex = isinstance(concatenated_stats.columns, pd.MultiIndex) and len(concatenated_stats.columns.levels) > 1
    #print(is_multiindex)

    mean = {
        "stat": concatenated_stats.mean(axis=1, level=1) if is_multiindex else concatenated_stats.mean(axis=1),
        "spread": concatenated_stats.std(axis=1, level=1) if is_multiindex else concatenated_stats.std(axis=1)
    }

    median = {
        "stat": concatenated_stats.median(axis=1, level=1) if is_multiindex else concatenated_stats.median(axis=1),
        "spread": concatenated_stats.groupby(axis=1, level=1, sort=False).apply(lambda group: group.quantile(0.75, axis=1) - group.quantile(0.25, axis=1)) if is_multiindex else (concatenated_stats.quantile(0.75, axis=1) - concatenated_stats.quantile(0.25, axis=1))
    }

    return mean, median

def file_path_factory(directory=None, given_name=None, file_name_str_list=None, add_datatime=True, extension=''):
    # Combine the optimizer name, file id, and file name into the filename
    if not given_name:
        if not file_name_str_list:
            file_name_str_list = [str(uuid.uuid4())[:8]]
        file_name_str = '_'.join(file_name_str_list)
    else:
        file_name_str = given_name
    if add_datatime:
        file_name_str += '_' + datetime.now().strftime("%Y%m%d%H%M%S")
    if extension:
        file_name_str += '.' + extension
    try:
        if directory:
            # Make sure the directory exists. If not, create it.
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = f"{directory}/{file_name_str}"
        else:
            file_path = f"{file_name_str}"
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
    return file_path