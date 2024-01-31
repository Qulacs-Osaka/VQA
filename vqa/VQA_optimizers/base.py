import os
from logging import getLogger, FileHandler, Formatter, DEBUG
import hashlib
import uuid
import dill
import numpy as np
import time
from math import ceil

from numpy.random import *
import copy
from copy import deepcopy
from collections.abc import Iterable
from copy import deepcopy
from abc import ABC, abstractmethod

def map_to_pi_range(x, mask_inds=None):
    if mask_inds is None:
        mask_inds = []
    y = np.array(x)  # Create a copy to avoid modifying the original array
    temp_y = y[mask_inds] % (2 * np.pi)  # Apply modulo operation only to selected indices
    temp_y -= 2 * np.pi * (temp_y > np.pi)  # Adjust values only for selected indices
    y[mask_inds] = temp_y  # Update the original array with the modified values
    return y

class CustomFormatter(Formatter):
    def formatTime(self, record, datefmt=None):
        print(record.created)
        return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(record.created))

def init_logging(unique_id, raw_id_for_filename=False, directory=None):
    '''
    unique_id のハッシュのファイル名のログファイルを作り、そこに記録するloggerオブジェクトを返す
    unique_id として直接、ログに記録したい変数などを渡しても良い。
    Return:
        logger
    '''
    if directory is None:
        directory = 'log_files'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if raw_id_for_filename:
        hash_id = unique_id
    else:
        hash_id = hashlib.md5(str(unique_id).encode()).hexdigest()
    log_filename = f"{directory}/opt_{hash_id}.log"
    logger = getLogger(str(hash_id))

    # 既存のハンドラーをクリア
    # ルートロガーがストリームロガーを持っていると全部プリントされるので削除
    root_logger = getLogger()
    for handler in root_logger.handlers:
        handler.close()
        root_logger.removeHandler(handler)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = FileHandler(log_filename)
    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)
    logger.info(f'logging gets started for: {unique_id}')
    return logger


def ffill_list(my_list):
    '''
    listのNoneを、直前のNoneでない値で埋める（破壊的）
    '''
    if (not isinstance(my_list, Iterable)) or isinstance(my_list, str):
        return
    last_valid = None
    for i in range(len(my_list)):
        if my_list[i] is None and last_valid is not None:
            my_list[i] = deepcopy(last_valid)
        elif my_list[i] is not None:
            last_valid = deepcopy(my_list[i])
            
def generate_power_rule(coeff, exp, offset, is_decreasing=True, should_return_int=False):
    if is_decreasing:
        exp *= -1

    def power_rule(t):
        result = coeff * (t) ** exp + offset
        if should_return_int:
            result = np.floor(result).astype(int)
        return result

    return power_rule

def generate_2p_fixed_power_rule(xmin, xmax, ymin, ymax, exp, horizontal_shift=True):
    '''
    ymax<ymin の場合（減少させたい場合）でもできる。
    '''
    if horizontal_shift:
        def power_rule(x):
            '''
            horizontal shift a(x+c)^b
            '''
            #ymax < ymin の場合、 x > xmax でexpの中が負になると、非整数expでinvalidになるから注意
            #print(f'ymin {ymin}, xmin {xmin}, ymax {ymax}, xmax {xmax}, exp {exp}')
            return ymin* np.maximum((x-xmin)/(xmax-xmin) * ((ymax/ymin)**(1/exp)-1.)+1., 1e-5) ** exp
        return power_rule
    else:
        def power_rule(x):
            '''
            vertical shift ax^b + c
            '''
            return np.maximum((ymax-ymin)*((x-xmin)/(xmax-xmin))**exp + ymin, 1e-6)
        return power_rule

def decr_Adam(t, params):
    '''
    Adam のハイパーパラメータ
    alpha_t = params[0] * (t + 1) ** (- params[1])
    beta1 = params[2]
    beta2 = params[3]
    として、bias correctionを、
    alpha_t * np.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)
    の形で入れたもの。
    実装法にのっとり、alpha_t, beta1, beta2 を返す（beta1, 2 はコンスタント）
    '''
    beta1 = params[3]
    beta2 = params[4]
    if t==0:
        alpha_t = params[0]
    else:
        alpha_t = (params[0] * (t + params[1]) ** (- params[2])) * np.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)
    return alpha_t, beta1, beta2


class hpara_rules:
    '''
    step sizeなどの一般的な更新規則として、tをとってhyper parameter(s) を返す関数を使いたいときに、このクラスを使う。
    paramsでspecifyされた更新規則の関数として使えるインスタンスを生成しつつ、params_listの値を保持する。
    例えば、alpha(t, [a, b])として、a * t**(-b) をステップサイズとして返す関数を使って、a, bについてサーチしたい場合、a, bのリストab_listに渡って、
    [hpara_rules(alpha, [a, b]) for a, b in ab_list] のように生成したリストを、grid_para['hpara_rule']に与えれば、a, bについてサーチできて、a,bの値も結果として参照できる
    （結果として、オブジェクトが返ってくるので、.params アトリビュートを参照すればよい。）
    '''
    def __init__(self, hpara_rule, hparams):
        self.hpara_rule = hpara_rule
        self.hparams = hparams
    
    def __call__(self, t):
        return self.hpara_rule(t, self.hparams)


class Optimization_analyzer:
    def __init__(self):
        pass

class VQA_optimizer_base(ABC):
    '''
    optimizerを作るときのベースのクラス。これを継承してoptimizerを作る
    variable_argsは、それぞれによって異なり、具体的なoptimizerクラスで定義されて、内部状態として処理される。
    最適化を継続する場合は、これも保存して読み込まないといけない。
    manual_continue_conditionがTrueのときは、hpara['continue_condition']が与えられればそれを評価する。
    Falseのときは、max_shots, max_time, max_iter いずれかを与えて、そのバウンドに達したら終了とする。複数が与えられたら、どれかが達したら終了。どれも与えられていなかったらmax_iterのdefaultにする。
    また、Falseのとき、hpara['continue_condition']を与えたら、それを追加の条件にすることもできる。（例えば、誤差がある値より大きいかつshot数が一定以下の限り続けるなど）
    これらの設定は、途中から最適化を続けたい場合などに、アトリビュートの変更で別の設定にすることができる。
    hparaの、continue_conditionは、callable(obj)で、これのreturnがTrueである限り、最適化のイテレーションを続ける
    これは基本的に１回の最適化の試行を特定のオプティマイザーでするためのクラス。複数の試行の結果は、results_list アトリビュートに保存できる。
    initializationを行うと次の要素に保存されるようになるので、上書きして消してしまうのを防止するのに使える。
    データの処理は専用のクラスを使う。
    '''
    def __init__(self):
                 #, hpara, init_params=None, target_obj=None, shot_time=1, per_circuit_overhead=0., communication_overhead=0., manual_continue_condition=False, func_to_track=None):
        ### hyper parameter initialization
        #self.starting_message = 'Optimizating ...'
        #superで呼ぶ
        self.results_list = []
        self._results_index = -1 #results_listにresultsを入れるインデックス。initializationが呼ばれる度にインクリメントすることで、続きから最適化をした場合は同じ要素が更新される。
        self.initialized = False
        #self.set_hyper_parameters(hpara, shot_time, per_circuit_overhead, communication_overhead, manual_continue_condition)
        ##### optimizer initialization
        #self.initialize_optimization(init_params, target_obj, func_to_track)
        
    ##########
    def set_hyper_parameters(self, hpara):
        '''
        上書きしてsuperで継承して必ず実行。
        通常のハイパーパラメータは、default_values (dict)で、キーとして変数名を、値でデフォルト値を指定、
        self._set_hpara(hpara, default_values)とすることでセットする。
        learning rate可変などで、ステップごとに呼び出す rule は、set_rule でセットする。
        デフォルトでは p[0]*(t+p[2])**p[1] をセットする。詳しくはset_rule のdocstring参照。
        shots_ruleと、batch_size_rule は共通で設定している。(hparaに与えられなくても問題ない)
        デフォルト
        hyper_parameter_values = {
            'SA_num': 10, #optimizeで、take_SA をoptionsで指定するときに使われる。
            'burnin_rate_SA': 0.9, #burninとしてSAをとらないiterationの割合
            'BMA_num': 10, #SAと同様
            'burnin_rate_BMA': 0.9, #
            'batch_size_rule': None,
            's_list_rule': None,
            'LR_rule': None,
            'LR_annealing_by_shots': True,
            'n_shots_annealing_by_shots': True,
            's_list_annealing_by_shots': True,
            'batch_size_annealing_by_shots': True,
            'norm_test_kappa_b': 0.9/np.sqrt(2),
            'norm_test_kappa_s': 0.9/np.sqrt(2),
            'norm_test_avg_num': 10,
            'norm_test_avg_gamma': 0.38,
            'shots_norm_test_on_avg': False
        }
        LR_rule
        n_shots_rule
        batch_size_rule
        それぞれ、hoge_ruleに対し、
        hoge_final_shots: 終端ショット数
        hoge_y0: shots = 0のときの値
        hoge_yfin: 終端ショット数での値
        hoge_exp: 指数(1に近いほど、直線的に増えて、大きいほど最初は遅く、最後に急激。１より小さいと逆。)
        hoge_horizontal_shift: Trueのときは、 a(x + c)^exp の２点固定で、Falseのときは、a x^exp + c を使う。
            違いは、後者では必ずx=0で微分0の曲線ということ。
        '''
        hyper_parameter_values = {
            'SA_num': 10, #optimizeで、take_SA をoptionsで指定するときに使われる。
            'burnin_rate_SA': 0.9, #burninとしてSAをとらないiterationの割合
            'BMA_num': 10, #SAと同様
            'burnin_rate_BMA': 0.9, #
            'batch_size_rule': None,
            's_list_rule': None,
            'LR_rule': None,
            'LR_annealing_by_shots': True,
            'n_shots_annealing_by_shots': True,
            's_list_annealing_by_shots': True,
            'batch_size_annealing_by_shots': True,
            #for mini-batch based norm test
            'norm_test_kappa_b': 0.9/np.sqrt(2),
            'norm_test_kappa_s': 0.9/np.sqrt(2),
            'norm_test_avg_num': 10,
            'norm_test_avg_gamma': 0.38,
            'shots_norm_test_on_avg': False,
            'apply_QEM': False,
            'QEM_warm_up_shots': 1e5, #とりあえずこっち
            'QEM_warm_up_iter': 10, #とりあえず不使用            
            'kwargs': {}
        }
        self._set_hpara(hpara, hyper_parameter_values)
        # LR_ruleの初期化
        self._set_default_power_rule(rule_name='LR_rule', 
                                    annealing_by_shots=self.LR_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=0.1, 
                                    yfin=1e-4, 
                                    exp=0.01, 
                                    final_shots=None, #あくまでデフォルト 無いとエラー。max_shotsはoptionで与えるのでここでデフォルト指定はできない
                                    coeff=0.1, 
                                    offset=1,
                                    is_decreasing=True)

        # n_shots_ruleの初期化
        self._set_default_power_rule(rule_name='n_shots_rule', 
                                    annealing_by_shots=self.n_shots_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=0, 
                                    yfin=1e5, 
                                    exp=1.3, 
                                    final_shots=None, 
                                    coeff=4, 
                                    offset=1,
                                    is_decreasing=False, 
                                    should_return_int=True)

        # s_list_ruleの初期化
        self._set_default_power_rule(rule_name='s_list_rule', 
                                    annealing_by_shots=self.s_list_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=None, 
                                    yfin=None, 
                                    exp=1.3, 
                                    final_shots=None, 
                                    coeff=None, 
                                    offset=None, 
                                    is_decreasing=False, 
                                    should_return_int=True)


        # batch_size_ruleの初期化
        self._set_default_power_rule(rule_name='batch_size_rule', 
                                    annealing_by_shots=self.batch_size_annealing_by_shots, 
                                    hpara=hpara, 
                                    y0=2, 
                                    yfin=64, 
                                    exp=1.3, 
                                    final_shots=None, 
                                    coeff=6, 
                                    offset=1, 
                                    is_decreasing=False, 
                                    should_return_int=True)


    
    @abstractmethod
    def initialize_optimization(self, init_spec_dict):
        '''
        最適化対象のオブジェクトを与えて、ショット数などのカウンターを渡す。再び初期化をしたいときに使う。必ず継承のときに呼び出す。
        具体的なクラスに継承するときはこれを上書きして、superで呼び出した後、cost funcやgradの関数を渡すなど、残りの初期化部分を記述する。
        '''
        pass
    
    def _fetch_n_params(self, init_spec_dict):
        '''
        n_paramsを、selfが持っていれば返し、なければ self.init_paramsまたは、self.target_objから取得
        それもなければ、init_spec_dictから取得し、self.n_paramsに入れつつ、returnする
        '''
        if hasattr(self, 'n_params'):
            n_params = self.n_params
            return n_params
        elif hasattr(self, 'init_params'):
            n_params = len(self.init_params)
        elif hasattr(self, 'target_obj') and hasattr(self.target_obj, 'n_params'):
            n_params = self.target_obj.n_params
        elif 'init_params' in init_spec_dict:
            n_params = len(init_spec_dict['init_params'])
        elif 'target_obj' in init_spec_dict:
            n_params = init_spec_dict['target_obj'].n_params
        else:
            #print('initialization has not completed because n_params is not given')
            return None
        self.n_params = n_params        
        return n_params
    
    def _basic_init_routine(self, init_spec_dict, variables_to_set, init_values_dict):
        """
        Optimization object's basic initialization routine.
        これを、initialization_optimization で呼び出す。variables_to_setとか init_values_dictをサブクラスに合わせて設定。
        サブクラス共通のものは含まれている。

        This function initializes the optimization object based on the given dictionaries, init_spec_dict, 
        variables_to_set, and init_values_dict. The function first initializes various counters used for track
        the optimization process. It then extracts target object, initial parameters, and the track function
        from init_spec_dict. If a target object and initial parameters are provided, it initializes specific
        attributes based on variables_to_set and init_values_dict.

        Parameters
        ----------
        init_spec_dict : dict
            Dictionary specifying initial settings. Expected to include keys like 'target_obj', 'init_params', 
            and 'func_to_track' and keys (typically indicating callable such as 'iEvaluate') specified by variables_to_set.

        variables_to_set : list
            List of strings specifying the names of the variables to be initialized other than common items (target_obj, init_params, func_to_track)
            例えば、iEvaluateなど、最適化アルゴリズムに依存してスペシファイしないといけないものの名前をここで与える。
            与えた名前の値は、init_spec_dictに同じ名前のキーとして与えるか、target_obj.methods_for_optimization 辞書に同様に与える。
            selfの同じ名前の属性に保持される。これを呼び出して optimizer_stepなどで使う。

        init_values_dict : dict
            Dictionary specifying the initial values of the attributes of the optimization object. The keys should 
            match the attributes of the object, and the values should specify the initial values.
        init_values_dict: 辞書型オプションで、最適化オブジェクトの属性とその初期値を定義します。
        キーはselfに保持される属性の名前（文字列）、値はその属性の初期値であり、以下のフォーマットのいずれかを持つことができます：
        - 任意の値: その値はそのまま属性に設定されます。
        - 'zeros': np.zerosが使用され、配列の長さは初期パラメータの数（n_params）と同じになります。
        - ('zeros', dtype): np.zerosが使用され、配列の長さはn_paramsと同じになりますが、dtypeは指定されたデータ型になります。
        - ('full', fill_value): np.fullが使用され、配列の長さはn_paramsと同じで、全ての要素がfill_valueで満たされます。
        - 文字列: 文字列はコードとして評価され、その評価結果が属性の値になります。この文字列内でvariables_to_setの項目名を使用すると、それらはinit_spec_dictから対応する属性の値に置き換えられます。
           例えば、'np.full(iEvaluate(a)*b)'という文字列があった場合、'iEvaluate'は対応するcallableに置き換えられ、'a'と'b'はそれぞれ対応する属性の値self.a, self.bに置き換えられます。
           selfの属性から参照するので、selfをつけてはいけないことに注意。
           n_paramsは、self.n_paramsを参照します。既にself.n_paramsがなければ、init_pramsやtarget_objから自動的に取得します。
           ただし、このコード文字列はわかりにくいので、極力つかわないほうがいい。
           n_paramsなどを使いたいときは、initialization_optimizationのコードに個別にn_paramsを取得する部分を書いた方がいい。
        例：Adam の場合、
        init_values_dict = {
            'm_t': 'zeros',
            'v_t': 'zeros'
        }
        を与える。

        M_tot: ターゲットオブジェクトに存在する場合、それはそのまま取得されます。存在しない場合は、ターゲットオブジェクトの 'x_train' 属性の長さとなります。何もなければ 0 になります。

        Returns
        -------
        None
        """
        #最初にx軸は一律で0の値を持たせて初期化しないと、統計処理をするとき問題になる。
        #logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.iteration_num = 0
        self.total_shots = [0]
        self.total_time = [0.]
        self.total_call = [0]
        self.total_epoch = [0]
        self.shots_per_step = [0]
        self.time_per_step = [0.]
        self.call_per_step = [0]

        target_obj = init_spec_dict.get('target_obj', getattr(self, 'target_obj', None))
        init_params = init_spec_dict.get('init_params', getattr(self, 'init_params', None))        

        if target_obj is not None:
            self.target_obj = target_obj
            if hasattr(target_obj, 'periodic_para_inds'):
                self.periodic_para_inds = target_obj.periodic_para_inds
            else:
                self.periodic_para_inds = np.arange(target_obj.n_c_params)
            self.shots_counter = getattr(target_obj, 'shots_counter', [0])
            self.call_counter = getattr(target_obj, 'call_counter', [0])
            self.circuit_counter = getattr(target_obj, 'circuit_counter', [0])
            #communication_counterだけはデフォルトNoneにして与えていないときは区別する。
            # (communicationのカウントは単に1stepで1回のことが多いため、その場合comm countをoptimizeで自動カウントする)
            self.communication_counter = getattr(target_obj, 'communication_counter', None)
            self.epoch_counter = getattr(target_obj, 'epoch_counter', [0])
            self.shot_time = getattr(target_obj, 'shot_time', 1.)
            self.per_circuit_overhead = getattr(target_obj, 'per_circuit_overhead', 0.)
            self.communication_overhead = getattr(target_obj, 'communication_overhead', 0.)
            n_params = self._fetch_n_params(init_spec_dict)
            variables_to_set.append('func_to_track')
            for var in variables_to_set:
                value = init_spec_dict.get(var, None)                
                if value is None and hasattr(self, 'target_obj') and var in getattr(self.target_obj, 'methods_for_optimization', {}):
                    value = self.target_obj.methods_for_optimization.get(var, None)                
                setattr(self, var, value if value is not None else getattr(self, var, None))
                #if getattr(self, var, None) is not None:
                    #print(f'{var} has been set.')
        func_to_track = init_spec_dict.get('func_to_track', getattr(self, 'func_to_track', None))
        if init_params is not None:
            self.init_params = init_params.copy()
            self.para_evol = [init_params.copy()]
            #print('init_params has been set.')
            #logging.debug('just after setting init_params')
            #print(self.para_evol)
            self.para_evol_SA = []
            self.func_to_track = func_to_track
            #logging.debug('just after setting func_to_track')
            #logging.debug('just before check if func_to_track is None')
            if func_to_track is None:
                #print('initialization has not completed because func_to_track is not given')
                return
            #logging.debug('just before referring func_to_track')
            #logging.debug(f'func to track {self.func_to_track}')
            #try:
                #並列化でなぜか失敗したポイント
                #logging.debug(f'calling func to track {self.func_to_track(init_params)}')
                #self.func_evol = [self.func_to_track(init_params)]
            self.func_evol = [self.func_to_track(init_params)]
            #except:
            #    logging.exception('func to track failed')
            #logging.debug('just after setting func_evol')
            self.func_evol_BMA = []
            self.func_evol_SA = []

            #if hasattr しないと、属性を持っていなかったらエラーになる。
            if hasattr(self, 'start_time'):
                del self.start_time

        #logging.debug('just before parsing init_values_dict')
        if target_obj is not None:
            for attr, init_val in init_values_dict.items():
                try:
                    if isinstance(init_val, str) and init_val == 'zeros':
                        #logging.debug(f'zeros {attr}: {init_val}')
                        setattr(self, attr, np.zeros(n_params))
                    elif isinstance(init_val, tuple) and init_val[0] == 'zeros':
                        if isinstance(init_val[1], type):
                            setattr(self, attr, np.zeros(n_params, dtype=init_val[1]))
                        else:
                            setattr(self, attr, np.zeros(n_params))
                    elif isinstance(init_val, tuple) and init_val[0] == 'full':
                        #logging.debug(f'full {attr}: {init_val}')
                        setattr(self, attr, np.full(n_params, init_val[1]))
                    elif isinstance(init_val, str): #code sequence including functions to be set by init_spec_dict
                        # Extract callable items from variables_to_set
                        #logging.debug(f'code string {attr}: {init_val}')
                        callables = {var: getattr(self, var) for var in variables_to_set if callable(getattr(self, var))}
                        # Replace callable items in init_val
                        for var, func in callables.items():
                            init_val = init_val.replace(var, f'{func.__name__}()')
                        # Evaluate the string as code
                        value = eval(init_val, globals(), vars(self))
                        #logging.debug('code excecuted')
                        setattr(self, attr, value)
                    #elif isinstance(init_val, tuple) and init_val[0] in variables_to_set and callable(getattr(self, init_val[0])):
                    #    func = getattr(self, init_val[0])
                    #    args = init_val[2] if len(init_val) > 2 else ()
                    #    kwargs = init_val[3] if len(init_val) > 3 else {}
                    #    setattr(self, attr, func(*args, **kwargs))
                    else:
                        #logging.debug(f'raw_value {attr}: {init_val}')
                        setattr(self, attr, init_val)
                except Exception as e:
                    print(f'Error while initializing attribute {attr}: {e}')

            self.M_tot = getattr(target_obj, 'num_train_data', len(getattr(target_obj, 'x_train', [])))
            #running averageとる用
            self.grad_evol = []
            #self.run_avg_count = 0

        variables_to_set.append('target_obj')
        variables_to_set.append('init_params')
        #print(variables_to_set)
        #print([getattr(self, var, None) for var in variables_to_set])
        if all(getattr(self, var, None) is not None for var in variables_to_set):
            try:
                self._finalize_initialization()
            except Exception as e:
                print(f'Error during finalization: {e}')
        else:
            self.initialized = False
            
    
    #initialize_optimizationで、適切に初期化できたときに必ず呼び出す。
    def _finalize_initialization(self):
        #logging.basicConfig(level=logging.DEBUG, filename='finalize_init.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.initialized = True
        self.file_id = uuid.uuid4() # Create a unique identifier
        self._results_index += 1
        self.results_list.append({})
        self.tracked_var_dict = {}
        self.last_recorded_point_dict = {}
        self._first_time = True #最初のイテレーションであることを示す。
        #logging.debug('before init epoch counter')
        if self.epoch_counter:
            self.epoch_counter[0] = 0
        #logging.debug('before delete attr')
        if hasattr(self, 'track_interval'):
            del self.track_interval
        if hasattr(self, 'track_interval_extra_var_dict'):
            del self.track_interval_extra_var_dict
        if hasattr(self, 'new_track_interval_extra_var_dict'):
            del self.new_track_interval_extra_var_dict
        #print('initialization completed.')
        
    def _set_hpara(self, hpara, default_values):
        '''
        keyとそのデフォルト値の辞書を使用して、まとめて属性を設定するメソッド。
        hyperparametersのセットでこれが使える部分はこれを使うべき。
        hpara: selfの属性にsetする値を入れた辞書。キーがそのままアトリビュート名になる。与えないものはデフォルトになる。
        default_values: セットするべきアトリビュート名をキーに、デフォルトの値をそれぞれの値とした辞書。
        ただし、hparaに与えなかったときの優先順位として、selfのすでに持っている値がデフォルト値よりも優先される。
        '''
        for key, default_value in default_values.items():
            setattr(self, key, hpara.get(key, getattr(self, key, default_value)))
    
    def _set_rule(self, hpara, rule_name, default_values, annealing_by_shots=True, generate_rule_function=None):
        '''
        パラメータの定義と規則関数の生成を行います。default_valuesには、generate_rule_functionの引数名と対応するhparaのキー、デフォルト値のタプルのリストを与えます。
        このメソッドは、指定された規則関数（デフォルトではgenerate_power_rule）を使用して、ルール（関数）を設定します。このルールは、最適化のステップ数に基づいてステップサイズなどを決定します。
        このメソッドは、具体的なオプティマイザのクラスを作るときに、set_hyper_parameters で呼び出して使う。
        '''
        kwargs_for_rule = {}
        if annealing_by_shots:
            if not hasattr(self, '_annealing_by_shots_rules'):
                self._annealing_by_shots_rules = []
            self._annealing_by_shots_rules.append(rule_name)
            if generate_rule_function is None:
                generate_rule_function = generate_2p_fixed_power_rule
        else:
            if generate_rule_function is None:
                generate_rule_function = generate_power_rule
        for param_name, hpara_key, default_value in default_values:
            if hpara_key is None:  # For parameters like is_decreasing and should_return_int that are fixed and not given in hpara
                kwargs_for_rule[param_name] = default_value
            else:
                value = hpara.get(hpara_key, getattr(self, hpara_key, default_value))
                setattr(self, hpara_key, value)
                kwargs_for_rule[param_name] = value
        given_rule = hpara.get(rule_name, None)
        if given_rule is not None:
            rule = given_rule
        elif any(k in hpara for _, k, _ in default_values if k is not None):
            rule = generate_rule_function(**kwargs_for_rule)
        else:
            rule = getattr(self, rule_name, None)
        if rule is None:
            rule = generate_rule_function(**kwargs_for_rule)
        setattr(self, rule_name, rule)

    def _set_default_power_rule(self, rule_name, annealing_by_shots, hpara, 
                               y0, yfin, exp, final_shots, init_shots=0, coeff=None, offset=None, 
                               is_decreasing=False, should_return_int=False):
        base_name = rule_name.replace('_rule', '')
        if annealing_by_shots:
            values = [
                ('xmin', f'{base_name}_init_shots', init_shots),
                ('xmax', f'{base_name}_final_shots', final_shots),
                ('ymin', f'{base_name}_y0', y0),
                ('ymax', f'{base_name}_yfin', yfin),
                ('exp', f'{base_name}_exp', exp),
                ('horizontal_shift', f'{base_name}_horizontal_shift', True)
            ]
            self._set_rule(hpara, rule_name, values, annealing_by_shots=True, generate_rule_function=generate_2p_fixed_power_rule)
        else:
            values = [
                ('coeff', f'{base_name}_coeff', coeff or 1.), 
                ('exp', f'{base_name}_exp', exp), 
                ('offset', f'{base_name}_offset', offset or 0.), 
                ('is_decreasing', None, is_decreasing),
                ('should_return_int', None, should_return_int)
            ]
            self._set_rule(hpara, rule_name, values, annealing_by_shots=False, generate_rule_function=generate_power_rule)

    def get_annealed_value(self, rule_name, t=None, s_tot=None, should_floor=False):
        '''
        shot数で step sizeの値をコントロールしていくか、iteration numberでコントロールしていくか、どっちでも対応できるように、このメソッドを通して、次の値を返す。
        self._annealing_by_shots_rules に登録されている rule_name のものは、ショット数で、それ以外はiteration numで決める。
        '''
        rule = getattr(self, rule_name)
        if rule_name in self._annealing_by_shots_rules:
            if s_tot is None:
                s_tot = self.total_shots[-1] + self.shots_counter[0]
            value = rule(s_tot)
        else:
            if t is None:
                t = self.iteration_num
            value = rule(t)

        if should_floor:
            if isinstance(value, np.ndarray):
                return np.floor(value).astype(int)
            else:
                return int(np.floor(value))
        return value

    def moving_average(self, estimator_current_dict, mu=0.9, t=None, bias_uncorrection_list=None, t_offset=0):
        '''
        tはbias correctionに使われる
        bias_uncorrection_list に入れたらbias correctionしない。それ以外はする。
        '''
        #print('modification')
        if bias_uncorrection_list is None:
            bias_uncorrection_list = []
        if t is None:
            t = self.iteration_num + 1
        updated_values = {}
        for key, value in estimator_current_dict.items():
            # 移動平均パラメータ名を作成
            moving_avg_param_name = f"{key}_avg"
            
            # self._first_time が True か、属性が存在しなければ適切な形で初期化
            if self._first_time or not hasattr(self, moving_avg_param_name):
                if isinstance(value, np.ndarray):  # valueがnumpyの配列かどうかを判断
                    moving_avg_value = np.zeros(value.shape)
                else:
                    moving_avg_value = 0
            else:  # それ以外の場合は、属性の値を取得
                moving_avg_value = getattr(self, moving_avg_param_name)
            
            # 移動平均で更新
            moving_avg_value = mu * moving_avg_value + (1-mu) * value
            setattr(self, moving_avg_param_name, moving_avg_value)  # 保存
            # bias correction
            if key in bias_uncorrection_list:
                #print(key)
                updated_values[key] = moving_avg_value #bias correctionなしー＞あり / (1 - mu ** t)
            else:
                updated_values[key] = moving_avg_value / (1 - mu ** (t - t_offset))                
        return updated_values
    
    def estimate_next_values(self, estimator_current_dict, method_function=None, **kwargs):
        '''
        iCANSやSanta など、次のステップでの勾配の値や分散を予測してショット数計算などに使うためのメソッド。
        method_function で指定して、
        estimator_current_dict には、予測変数名をキーとして、現在の値をいれた辞書を与える。
        例えば、gradから予測して、gradというキーにしたければ、{'grad':grad} という具合に。
        method_functionのreturn は、推定された値が入った辞書を想定。kwargsに、パラメータを入れる。
        デフォルトでは self.moving_averageを使用。引数は、平均パラメータmuと、現ステップ数 t。
        '''
        if method_function is None:
            method_function = self.moving_average
        return method_function(estimator_current_dict, **kwargs)
    
    ## file save utilities
    def save_to_file(self, results=None, save_self_obj=True, directory=None):
        '''
        一回の最適化が終わったときに、resultsまたは自身のオブジェクトを保存する
        '''
        # Combine the optimizer name, file id, and file name into the filename
        try:
            if directory:
                # Make sure the directory exists. If not, create it.
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = f"{directory}/{self.optimizer_name}_single_{self.file_name_str}_{self.file_id}.pkl"
            else:
                file_name = f"{self.optimizer_name}_single_{self.file_name_str}_{self.file_id}.pkl"
        except Exception as e:
            print(f"An error occurred while saving data: {e}")

        data_to_save = {}

        try:
            if save_self_obj:
                # Here we're making a shallow copy of self to avoid recursive reference when saving self.
                self_copy = copy.copy(self)
                # Removing reference to self to avoid infinite recursion.
                if hasattr(self_copy, 'self'): 
                    delattr(self_copy, 'self')
                # Removing reference to target_obj object to avoid pickle error
                # たまにmainで定義されたクラスを使うこともあるので、その場合にシリアライズに問題が生じることを防ぐにはこれを使うのもあり。
                #　もしシリアライズできるならそのまま使えばいいので、とりあえずそのまま保存する。
                #if hasattr(self_copy, 'target_obj'): 
                #    delattr(self_copy, 'target_obj')
                data_to_save["object"] = self_copy
            else:
                data_to_save["results"] = results

            with open(file_name, 'wb') as f:
                dill.dump(data_to_save, f)
            print(f"Data saved to file: {file_name}")
        except Exception as e:
            if save_self_obj:
                print(f"An error occurred while saving self object: {e}")
                print("Falling back to saving results instead.")
                self.save_to_file(results=results, save_self_obj=False, directory=directory)
            else:
                print(f"An error occurred while saving data: {e}")

    
    def find_results_files(self, directory=".", search_string=None):
        '''
        directory (str, optional): The directory where the file is located. 
                                   The default is the current directory (".").
                                   If the file is in a subdirectory of the current directory, 
                                   provide the subdirectory's name (e.g., "subdir/").
                                   If the file is in a completely different path,
                                   provide the full path (e.g., "/home/user/docs/").
        '''
        if search_string is None:
            search_string = f"{self.optimizer_name}_single"  # Default search string
        # List all files in the specified directory
        files_in_directory = os.listdir(directory)
        # Filter files based on the search string
        filtered_files = [file for file in files_in_directory if search_string in file]
        # Print the filtered files
        for file in filtered_files:
            print(file)
        return filtered_files
            
    def load_from_file(self, filename, directory="."):
        """
        Load data from a specified file and update the instance state.

        This method attempts to load data from a pickle file located at the
        specified directory. If the saved data is an object, it updates the
        current instance's state. If the saved data is 'results', it stores
        the results in the instance's 'latest_results' attribute. 
        The operation performed (self updated or results loaded) is printed out.

        Parameters:
        filename (str): The name of the file to load data from.
        directory (str, optional): The directory where the file is located. 
                                   The default is the current directory (".").
                                   If the file is in a subdirectory of the current directory, 
                                   provide the subdirectory's name (e.g., "subdir/").
                                   If the file is in a completely different path,
                                   provide the full path (e.g., "/home/user/docs/").

        Raises:
        Exception: If an error occurs while loading data from the file.
        """
        file_path = os.path.join(directory, filename)
    
        try:
            with open(file_path, 'rb') as f:
                loaded_data = dill.load(f)
            if "object" in loaded_data:
                self.__dict__.update(loaded_data["object"].__dict__)
                print(f"Self object updated from the file: {filename}")
            elif "results" in loaded_data:
                self.latest_results = loaded_data["results"]
                print(f"Results loaded into 'latest_results' from the file: {filename}")
        except Exception as e:
            print(f"An error occurred while loading data from file: {e}")
    
    @classmethod
    def load_multiple_objects(cls, filenames, directory="."):
        """
        Loads multiple instances from a list of files.

        This method attempts to load multiple instances from a list of pickle files 
        located at the specified directory. Each filename corresponds to an individual 
        instance to be loaded.

        Parameters:
        filenames (list of str): The list of filenames to load instances from.
        directory (str, optional): The directory where the files are located. 
                                   The default is the current directory (".").
                                   If the files are in a subdirectory of the current directory, 
                                   provide the subdirectory's name (e.g., "subdir/").
                                   If the files are in a completely different path,
                                   provide the full path (e.g., "/home/user/docs/").

        Returns:
        list: A list of loaded instances.
        """

        instances = []
        for filename in filenames:
            instance = cls()
            instance.load_from_file(filename, directory)
            instances.append(instance)
        return instances
        
    def _print_Monitor_variables(self, caller_locals=None):
        if self.Monitor_iter and (self.iteration_num % (self.Monitor_interval)) == 0:
            variable_names = self.Monitor_var_names
            if not variable_names:
                return

            for var_name in variable_names:
                value = getattr(self, var_name, None)
                if value is None and caller_locals is not None:
                    value = caller_locals.get(var_name, None)

                if value is not None:
                    if var_name == 'func_evol':
                        value = next((x for x in reversed(value) if x is not None), None)
                        log_msg = f'Last tracked {var_name}: {value}'
                    else:
                        log_msg = f'{var_name}: {value}'

                    if self.Monitor_by_logging:
                        self.logger.info(log_msg)
                    else:
                        print(log_msg)

            log_msgs = [
                f"iteration_num: {self.iteration_num}",
                f"current total shots: {self.total_shots[-1]:.4e}",
                f"current total time: {self.total_time[-1]:.4e}",
                f"elapsed_time: {time.time() - self.start_time}"
            ]

            if self.Monitor_by_logging:
                for log_msg in log_msgs:
                    self.logger.info(log_msg)
            else:
                for log_msg in log_msgs:
                    print(log_msg)

    def _track_variables(self, caller_locals=None):
        '''
        自身の属性か、caller_localsに、self.track_var_namesに名前のある場合、それをself.var_track_dictの該当項目に入っているリストに加える。
        caller_localsには、呼び出したところでのlocals()を渡すなどして使う。
        これはoptimizer_step内で呼ばれるべきである。
        これは、Santaのような、途中で最適化のフェーズが切り替わる場合にも、各フェーズで記録したい変数全部を名前のリストに入れておけばOKで、そのまま使えばいい。
        変数が存在しなければ、バイパスするようになっているからである。
        '''
        if self.track_var_names is None:
            return

        for var in self.track_var_names:
            value = getattr(self, var, None)
            if value is None and caller_locals is not None:
                value = caller_locals.get(var, None)
            if value is not None:
                # Default track interval for this variable
                track_interval = self.track_interval_extra_var_dict.get(var, self.track_interval)
                if self.track_indicator == 'iteration_num':
                    if self.iteration_num % track_interval == 0:
                        self.tracked_var_dict[var].append(value)
                    else:
                        self.tracked_var_dict[var].append(None)

                elif self.track_indicator == 'total_shots':
                    if var not in self.last_recorded_point_dict:
                        self.tracked_var_dict[var].append(value)
                        self.last_recorded_point_dict[var] = self.total_shots[-1]
                    else:
                        if self.total_shots[-1] - self.last_recorded_point_dict[var] >= track_interval:
                            self.tracked_var_dict[var].append(value)
                            self.last_recorded_point_dict[var] = self.total_shots[-1]
                        else:
                            self.tracked_var_dict[var].append(None)

                elif self.track_indicator == 'total_time':
                    if var not in self.last_recorded_point_dict:
                        self.tracked_var_dict[var].append(value)
                        self.last_recorded_point_dict[var] = self.total_time[-1]
                    else:
                        if self.total_time[-1] - self.last_recorded_point_dict[var] >= track_interval:
                            self.tracked_var_dict[var].append(value)
                            self.last_recorded_point_dict[var] = self.total_time[-1]
                        else:
                            self.tracked_var_dict[var].append(None)

    def minibatch_norm_test(self, estimate_next_dict, shots_norm_test_on_avg=False, batch_size=None,
                            r=10, gamma=0.38, kappa_b=0.9/np.sqrt(2), kappa_s=0.9/np.sqrt(2),
                            R_list=None, batch_size_min=2, batch_size_max=1e3, n_min=2, eps=1e-10):
        """
        Determines the batch size and shots for gradient vector estimation based on norm test.
        バッチサイズをdata決めた条件付き期待値の分散を使ったnorm testで、
        ショット数を、条件付き分散のdataについての平均を使って、normとして、gradのノルムまたは、data点条件付きでのノルムの平均を使って（shots_norm_on_avgで選ぶ）norm testで決める。
        イテレーションの途中で呼び出して使う。

        Args:
            self (object): The instance of the class containing this method.
            estimate_next_dict:
                grad (numpy array): The gradient vector to be estimated.
                gvar_list_EV (list): Estimated expectation values of the variance of each component of the gradient when data is fixed.
                gvar_list_b (list): Estimated variance of the mean of each component of the gradient.
            r (int): The length of the gradient evolution list `self.grad_evol`, determining the running average length for grad.
            kappa_b (float): A hyperparameter representing the standard for the norm test in batch size calculation (the allowable ratio of the expected norm deviation of grad's estimate to the norm of grad).
            kappa_s (float): A hyperparameter representing the standard for the norm test in shots calculation.
            gamma (float): A threshold factor to determine whether to replace grad with g_avg.
            batch_size_min (int): The minimum allowable batch size.
            R_list (list): A list of ratios representing the variance of each gradient component relative to the variance of the 0th component.
            n_min (int): The minimum allowable number of shots.

        Returns:
            batch_size (int): The determined batch size.
            n_list (numpy array): A list containing the determined number of shots for each component of the gradient.
        """
        logger = init_logging('norm_test', raw_id_for_filename=True)
        grad = estimate_next_dict['grad']
        gvar_list_EV = estimate_next_dict['gvar_list_EV']
        gvar_list_b = estimate_next_dict['gvar_list_b']
        norm = lambda x: np.linalg.norm(x)
        update_batch_size = False
        if batch_size is None:
            batch_size = self.batch_size

        self.grad_evol.append(grad.copy())

        if len(self.grad_evol) == r + 1:
            self.grad_evol.pop(0)
            
        temp_grad = grad
        if len(self.grad_evol) == r:
            g_avg = np.mean(self.grad_evol, axis=0)
            temp_grad = g_avg if norm(g_avg) < gamma * norm(grad) else grad

        # バッチサイズの決定
        M_tot = self.M_tot
        if not self.manual_batch_size:            
            if (M_tot - batch_size) / (M_tot - 1) * sum(gvar_list_b) / batch_size > kappa_b ** 2 * norm(temp_grad) ** 2:
                batch_size = max((M_tot - batch_size) / (M_tot - 1) * sum(gvar_list_b) / (kappa_b ** 2 * norm(temp_grad) ** 2 + eps), batch_size_min)
                logger.debug(f'norm(temp_grad)**2: {norm(temp_grad)**2}')
                logger.debug(f'gvar_list_b :{gvar_list_b}')
                logger.debug(f'batch_size :{batch_size}')
                batch_size = ceil(min(batch_size, batch_size_max))
                update_batch_size = True

            if update_batch_size:
                self.grad_evol = []
        else:
            if self.batch_size_annealing_by_shots:
                batch_size = self.batch_size_rule(self.total_shots[-1] + self.shots_counter[0])
            else:
                batch_size = self.batch_size_rule(self.iteration_num+1)

        # ショット数の決定
        V_0 = gvar_list_EV[0]
        if shots_norm_test_on_avg:
            g_norm2 = norm(temp_grad)**2 + sum(gvar_list_b) #=\sum_j E_Y[E[X_j|Y]^2]
        else:
            g_norm2 = norm(temp_grad)**2
        logger.debug(f'V_0 :{V_0}')
        logger.debug(f'g_norm2 :{g_norm2}')
        n_list_0 = ceil(max(sum(R_list) * V_0 / (kappa_s** 2 * g_norm2 * batch_size + eps), n_min))
        n_list = np.maximum(n_list_0 * gvar_list_EV / (V_0 * R_list + eps), n_min)
        n_list = np.ceil(n_list).astype(int)
        logger.debug(f'n_list :{n_list}')

        return batch_size, n_list
    
    def norm_test(self, estimate_next_dict, shots_norm_test_on_avg=False,
                    r=10, gamma=0.38, kappa_s=0.9/np.sqrt(2),
                    R_list=None, n_min=2, eps=1e-10):
        '''
        とりあえず使う予定ないので保留
        '''
        pass

    def _optimizer_preprocess(self):
        '''
        オーバーライドして、optimizationの前に呼び出したい処理を定義する
        '''
        pass

    def _optimizer_postprocess(self):
        '''
        オーバーライドして、optimizationのあとに呼び出したい処理を定義する
        '''
        pass
        
    @abstractmethod
    def optimizer_step(self, params, **kwargs):
        pass
    
    def new_optimize(self, init_spec_dict=None, n_params=None, init_low=-np.pi, init_up=np.pi, hpara=None, options=None, **kwargs):
        #logging.basicConfig(level=logging.DEBUG, filename='new_opt.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        if init_spec_dict is None:
            init_spec_dict = {}
        target_obj = init_spec_dict.get('target_obj')
        if getattr(self, 'target_obj', None) is None and target_obj is None:
            print('target_obj has not been given')
            return
        if getattr(self, 'init_params', None) is None and init_spec_dict.get('init_params') is None:
            if n_params is None:
                n_params = getattr(self.target_obj, 'n_params', None)
            try:
                init_spec_dict['init_params'] = np.random.uniform(init_low, init_up, n_params)
            except Exception as e:
                print(f"An error occurred: {e}")
                return
        if hpara is not None:
            self.set_hyper_parameters(hpara)
        ### initialize optimization (初期値など与えた場合)
        if init_spec_dict:
            self.initialize_optimization(init_spec_dict)
        #logging.debug('after initialization')
        return self.optimize(options=options, **kwargs)        

    def continue_optimize(self, options_to_update=None, kwargs_to_update=None):
        """
        Continue the optimization process from where it left off.
        By default, it uses the settings from the previous call to optimize. 
        However, these settings can be changed by providing them in the options_to_update dictionary.
        """
        # Prepare kwargs by updating the existing values with the new ones
        kwargs = getattr(self, 'kwargs', {}) # Assuming you have self.kwargs to store previous kwargs
        if kwargs_to_update:
            kwargs.update(kwargs_to_update)
        # Continue the optimization
        return self.optimize(options=options_to_update, **kwargs)
        
    def optimize(self, options=None, **kwargs):
        '''
        BMAとsuffix avg は、最適化途中ではとらず、あとから専用メソッドで para_evolや func_evol から計算するようにした。
        ちなみにsuffix avg は、averageをとった点を次のパラメータにとるのではなく、最適化が終わったあとに、得られたパラメータの数ステップの平均をとる。
        track_lossがTrueのときは、self.func_to_trackを呼び出し、その値の推移を記録する。
            基本的には、initialize_optimizationを呼び出して初期化した後に使うか、前の状態のまま続きからやるという使い方を推奨する。
        track_var_names は、ここに指定した名前の、最適化の途中で使われる変数はトラッキングされて、辞書に格納する。
        func_to_trackは、exactを与えることが多いが、noisyな場合は、すでに計算したパラメータでの値はそのまま使いたい
            そのためには、関数を定義するクラスで、設定した個数のcacheを保持して、cacheにあるパラメータで呼ばれたらその値を返すように実装すること
        track_interval間隔のみ、ロス関数を計算してevolを記録するが、BMA_numが2以上で、モデル平均を取る場合、
        results_opt_ind: resultsの'option_val'に、var_argsの、results_opt_indの値のリストを入れる
        optimizer_stepは、optimizer_step(params, **kwargs)の形で、new_paramsを返すか、
        ステップごとに変化するメタパラメータをもつ場合は、optimizer_step(params, variable_args, **kwargs)で、variable_argsはリスト、new_paramsと、variable_argsを返す
        shots_bound は、stepをやめる判定をするだけなので、一般には少し超えたショット数を使って終了する。
        iter_num を与えたら、そっちを優先してそのiter_numだけ繰り返す。
        communication数が可変のものは、communication_counterを与えて、最適化ステップの関数でカウントする（最適化ステップにも同じものを与える必要がある）。
        最適化ステップの定義のセルで作って定義の中で呼び出してしまえばいい。
        **kwargs として、最適化に使う関数や、gradをキーワードで渡す。
        '''
        #Monitor var と track var をプリントしたり記録したりするメソッドは、ここではなく、optimizer_step で呼び出される前提
        #continueする時用に、設定を属性に保存
        if options is None:
            options = {}
        options_key_default = {
            'manual_continue_condition': False, #最適化を続ける条件の指定 manualで与えたら、それだけを条件とする。
            'continue_condition': None, #以下はmanualなしの場合。continue_conditionを与えたら、max_hoge以下で、さらにそれを満たすときに続行。
            'max_shots': None,
            'max_iter': None,            
            'max_time': None,
            'max_epoch': None,
            'file_name_str': '', #保存するときのファイル名につける文字列
            'Monitor_var_names': None,
            'Monitor_iter': False, #Monitor_var_namesが与えられていても、Falseにすればプリントしない
            'Monitor_interval': 10,
            'Monitor_func_evol': True, #func_evolをモニターすることが多いので、namesにいちいち加えなくてもよくする
            'Monitor_by_logging': True, #printではなく、ログファイルにアウトプット
            'track_var_names': None,
            'track_indicator': "total_shots",
            'track_interval': 1,
            'track_interval_extra_var_dict': {}, #track_var_namesにしていした追加変数の記録間隔。Noneならtrack_intervalと同じ。
            'new_track_interval': None,
            'new_track_interval_extra_var_dict': {}, #
            'track_interval_change_point': None,
            'track_change_indicator': 'total_shots', # キー名に注意
            'track_loss': True,
            'track_para_evol': True,
            'store_results_in_target_obj': False,
            'store_results_in_self': False,
            'save_directory': None,
            'save_self_obj': True,
            'auto_save': False,
            'take_SA': True, #suffix averageをとった結果もresultsに含めるか
            'take_BMA': True, #BMAをとった結果もresultsに含めるか
            'shot_time': 1., #overhead時間をあとから指定することも可能。
            'per_circuit_overhead': 0.,
            'communication_overhead': 0.,
            'ffill_results': True, #resultsの各項目のリストがNoneを含む場合、直前のNoneでない値で埋める
            'process_id': 0 #並列処理のときのログファイルの制御などのためのプロセスid
        }
        #self.func_evol[0] = self.func_to_track(self.init_params)
        self._set_hpara(options, options_key_default) #_set_hpara では、optionsがキーを持っていない要素は、selfのすでに持っている値が最優先、次にデフォルト値になる
        #optimizationの仮定でtrackする変数の名前のリストを与えるとそれらを記録する辞書を与える。
        if self.Monitor_func_evol:
            if not self.Monitor_var_names:
                self.Monitor_var_names = {'func_evol'}
            else:
                self.Monitor_var_names.add('func_evol')
        if not hasattr(self, 'tracked_var_dict'):
            self.tracked_var_dict = {}
        for var in self.track_var_names:
            if var not in self.tracked_var_dict:
                self.tracked_var_dict[var] = []
        #記録間隔の辞書を初期化
        for var_name in self.track_var_names:
            # For self.track_interval_extra_var_dict
            if var_name not in self.track_interval_extra_var_dict or self.track_interval_extra_var_dict[var_name] is None:
                self.track_interval_extra_var_dict[var_name] = self.track_interval
            # For self.new_track_interval_extra_var_dict
            if var_name not in self.new_track_interval_extra_var_dict or self.new_track_interval_extra_var_dict[var_name] is None:
                self.new_track_interval_extra_var_dict[var_name] = self.new_track_interval
        self.kwargs.update(kwargs)
        #各記録変数が最後に記録されたショットや時刻を記録する辞書の初期化
        if not hasattr(self, 'last_recorded_point_dict'):
            self.last_recorded_point_dict = {}
        if 'func_evol' not in self.last_recorded_point_dict:
            # Initialize such that the condition is met at the first step
            self.last_recorded_point_dict['func_evol'] = - self.track_interval
        for var_name in self.track_var_names:
            if var_name not in self.last_recorded_point_dict:
                self.last_recorded_point_dict[var_name] = - self.track_interval_extra_var_dict[var_name]
        #self.avg_burnin = avg_burnin
        #self.BMA_num = BMA_num
        if not self.initialized:
            print('Initialization has not appropriately done. Do initialize_optimziation first.')
            return
        self.logger = init_logging(self.process_id)
            #self.initialize_optimization(init_params, target_obj)
        if self.manual_continue_condition:
            continue_condition = self.continue_condition
        else:
            if self.max_iter is None and self.max_shots is None and self.max_time is None and self.max_epoch is None:
                self.max_shots = 1e7
            def max_budget_condition(obj):
                spent_list = [obj.iteration_num, obj.total_shots[-1], obj.total_time[-1], obj.total_epoch[-1]]
                max_budget_list = [obj.max_iter, obj.max_shots, obj.max_time, obj.max_epoch]
                other_condition = self.continue_condition
                return all(b is None or s < b for s, b in zip(spent_list, max_budget_list)) and (other_condition is None or other_condition(obj))
            continue_condition = max_budget_condition
        comm_count_flag = 1
        if self.communication_counter is None:
            comm_count_flag = 0
            self.communication_counter = [0]
        new_params = self.para_evol[-1]
        self.start_time = getattr(self, 'start_time', time.time()) #もし続きからの場合は、前の開始時刻を引き継ぐ。
        print(f'{self.starting_message} --------------->')
        interval_changed = False
        if self.take_SA:
            self.track_para_evol = True
        if (not self.track_para_evol) or self.take_BMA:
            #para_evol をトラックしないならlossをトラックしないと意味ない
            self.track_loss = True
            
        self._optimizer_preprocess()
        #logging.debug('started')
        if self.apply_QEM:
            self.kwargs.update({'use_QEM': False}) #QEMを使う場合、optimizer_stepで使うロス関数が必ずこのキーワードをもっていないといけない
        while continue_condition(self):
            #logging.debug(f'{self.iteration_num} iteration')
            if not interval_changed and self.track_interval_change_point is not None and self.new_track_interval is not None:
                if (self.track_change_indicator == "total_shots" and self.total_shots[-1] >= self.track_interval_change_point) or \
                   (self.track_change_indicator == "iteration_num" and self.iteration_num >= self.track_interval_change_point) or \
                   (self.track_change_indicator == "total_time" and self.total_time[-1] >= self.track_interval_change_point):
                    self.track_interval = self.new_track_interval
                    self.track_interval_extra_var_dict = self.new_track_interval_extra_var_dict.copy()
                    interval_changed = True
            #counterをリセット。これらは最適化で呼び出される関数と紐づけられていて、呼び出されると自動でカウントされる前提。
            self.circuit_counter[0] = 0
            self.shots_counter[0] = 0
            self.call_counter[0] = 0
            self.communication_counter[0] = 0
            if (comm_count_flag == 0):
                self.communication_counter[0] = 1
            if self.apply_QEM and self.total_shots[-1] > self.QEM_warm_up_shots:
                self.kwargs['use_QEM'] = True
            #if self.apply_QEM and self.iteration_num > self.QEM_warm_up_iter:
            #    self.kwargs['use_QEM'] = True
            new_params = self.optimizer_step(new_params, **self.kwargs)
            new_params = map_to_pi_range(new_params, self.periodic_para_inds)
            self.iteration_num += 1
            ## track で記録しないときは、Noneでパディングする
            if self.track_para_evol:
                self.para_evol.append(new_params)
            else:
                self.para_evol.append(None)
            if self.track_loss:
                if self.track_indicator == 'iteration_num':
                    if self.iteration_num % self.track_interval == 0:
                        self.func_evol.append(self.func_to_track(new_params))
                    else:
                        self.func_evol.append(None)
                elif self.track_indicator == 'total_shots':
                    # Check if the total_shots has increased by track_interval or more since the last time function was executed
                    if self.total_shots[-1] - self.last_recorded_point_dict['func_evol'] >= self.track_interval:
                        self.func_evol.append(self.func_to_track(new_params))
                        # Update last_recorded_shots with the current total_shots
                        self.last_recorded_point_dict['func_evol'] = self.total_shots[-1]
                    else:
                        self.func_evol.append(None)
                elif self.track_indicator == 'total_time':
                    # Check if the total_time has increased by track_interval or more since the last time function was executed
                    if self.total_time[-1] - self.last_recorded_point_dict['func_evol'] >= self.track_interval:
                        self.func_evol.append(self.func_to_track(new_params))
                        # Update last_recorded_time with the current total_time
                        self.last_recorded_point_dict['func_evol'] = self.total_time[-1]
                    else:
                        self.func_evol.append(None)
                #
                #if self.BMA_num > 1:
                #    if self.iteration_num >= avg_burnin:
                #        self.func_evol_no_avg.append(self.func_to_track(new_params))
                #        self.func_evol.append(moving_average(self.func_evol_no_avg, self.BMA_num))
                #    elif (self.iteration_num > avg_burnin - self.BMA_num) or (self.iteration_num % self.track_interval == 0):
                #        self.func_evol_no_avg.append(self.func_to_track(new_params))
                #        self.func_evol.append(self.func_evol_no_avg[-1])
                #    else:
                #        self.func_evol.append(None)
                #elif self.iteration_num % self.track_interval == 0:
                #    self.func_evol.append(self.func_to_track(new_params))
                #else:
                #    self.func_evol.append(None)
            else:
                self.func_evol.append(None)
            #今回の最適化ステップで呼び出された関数で使用された各種のカウントが記録されている。
            shots_step = self.shots_counter[0]
            time_step = self.per_circuit_overhead * self.circuit_counter[0] + self.shot_time * shots_step + self.communication_overhead * self.communication_counter[0]
            call_step = self.call_counter[0]
            self.shots_per_step.append(shots_step)
            self.time_per_step.append(time_step)
            self.call_per_step.append(call_step)
            self.total_shots.append(self.total_shots[-1] + shots_step)
            self.total_time.append(self.total_time[-1] + time_step)
            self.total_call.append(self.total_call[-1] + call_step)
            self.total_epoch.append(self.epoch_counter[0])
            #if self.Monitor_iter and (self.iteration_num % (self.Monitor_interval))==0:
            #    print(self.iteration_num)
            #    # Get the last non-None value in the list
            #    last_non_none_value = next((x for x in reversed(self.func_evol) if x is not None), None)
            #    print("current loss value:", last_non_none_value)
            #    print("current total shots: {0:.4e}".format(self.total_shots[-1]))
            #    print("current total time: {0:.4e}".format(self.total_time[-1]))
            #    print("elapsed_time:", time.time() - self.start_time)
            if self._first_time:
                self._first_time = False
        if self.func_evol[-1] is None:
            self.func_evol[-1] = self.func_to_track(self.para_evol[-1])
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.logger.info('Optimization process finished.')
        self.logger.info(f'total iteration {self.iteration_num}')
        self.logger.info(f'elapsed time for optimization: {self.elapsed_time}')
        print("total iteration:", self.iteration_num)
        print("elapsed time for optimization:", self.elapsed_time)
        self.circuit_counter[0] = 0
        self.shots_counter[0] = 0
        self.communication_counter[0] = 0
        #この実装では、resultsとして返していたものはすべてアトリビュートとして保持している。
        #しかし、複数の試行のresultsを得たい場合、resultsのリストに保存しないといけない。これをオプティマイザオブジェクトたちのリストにするのは違う気がする。
        #そこで、やはり必要なresultsにあたるアトリビュートを返す必要がある
        ###基本の記録すべき変数をresultsとして返す。
        #SA and BMA
        results = {
            "iteration_number": np.arange(self.iteration_num+1),
            "para_evol": self.para_evol, 
            "func_evol": self.func_evol, 
            "total_shots": self.total_shots, 
            "total_time": self.total_time, 
            "total_call": self.total_call,
            "total_epoch": self.total_epoch,
            "shots_per_step": self.shots_per_step, 
            "time_per_step": self.time_per_step, 
            "call_per_step": self.call_per_step
        }
        results.update(self.tracked_var_dict)
        # 削除条件を満たすキーのリストを作成
        keys_to_remove = [key for key, val in results.items() 
                        if isinstance(val, Iterable) and not isinstance(val, str) 
                        and all(item is None for item in val)]
        # 該当するキーを削除
        for key in keys_to_remove:
            del results[key]
        #results_df = pd.DataFrame(self.results_dict)
        self.latest_results = results.copy()
        if self.store_results_in_self:
            self.results_list[self._results_index] = results.copy()
        if self.store_results_in_target_obj:
            key_string = self.optimizer_name
            try:
                # check if self.target_obj exists
                if hasattr(self, 'target_obj'):
                    # check if results_dict attribute exists and is a dictionary
                    if hasattr(self.target_obj, 'results_dict') and isinstance(self.target_obj.results_dict, dict):
                        if key_string in self.target_obj.results_dict:
                            # append the results to the list
                            self.target_obj.results_dict[key_string].append(results.copy())
                        else:
                            # if key is not in the dictionary, add the key and initialize the list with results
                            self.target_obj.results_dict[key_string] = [results.copy()]
                    else:
                        # if results_dict does not exist or is not a dictionary, create a new dictionary
                        self.target_obj.results_dict = {key_string: [results.copy()]}
                else:
                    print("self.target_obj does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")
        #SA and BMA これらは、self.latest_resultsに直接加えるメソッド
        if self.take_SA:
            self.calculate_SA(para_evol=self.para_evol, SA_num=self.SA_num, 
                              func_evol=self.func_evol, burnin_rate=self.burnin_rate_SA, func_to_track=self.func_to_track, store_results_in_self=self.store_results_in_self)
        if self.take_BMA:
            self.calculate_BMA(func_evol=self.func_evol, BMA_num=self.BMA_num, burnin_rate=self.burnin_rate_BMA, store_results_in_self=self.store_results_in_self)
        if self.auto_save:
            self.save_to_file(results=results, save_self_obj=self.save_self_obj, directory=self.save_directory)
        if self.ffill_results:
            for value in self.latest_results.values():
                ffill_list(value)
        self._optimizer_postprocess()
        results = self.latest_results.copy()
        self.logger.info('Optimization done.')
        self.logger.info('-----------------------------------')
        return results
    
    #suffix を 1に設定すれば、そのまま para_evolからfunc_evolを計算する
    def calculate_SA(self, para_evol=None, SA_num=10, func_evol=None, burnin_rate=0.9, func_to_track=None, store_results_in_self=False):
        """
        Calculates the suffix average of `func_to_track` for each step in `para_evol`, taking into account the last `SA_num` parameters.
        If `SA_num` is larger than the available parameters, the function uses as many as available for the average.
        It then appends the result to `self.func_evol_SA` and to the 'func_evol_SA' element of `self.latest_results`.

        Parameters:
        - para_evol (list): The evolution of parameters in a previous optimization process. If None, `self.para_evol` will be used. If that's also None, the function ends with a message.
        - SA_num (int): The number of last parameters to consider for the average. Default is 10.
        - func_evol (list): The evolution of a function. If None, `self.func_evol` will be used. If that's also None, `func_to_track` will be calculated for each parameter set in `para_evol`.
        - burnin_rate (float): Fraction of total iterations after which the suffix averaging begins. Default is 0.9.
        - func_to_track (function): The function to calculate for each parameter set. If None, `self.func_to_track` will be used. If that's also None, the function ends with a message.
        - store_results_in_self (bool): Whether to store the results in `self.results_list[self._results_index]`. Default is True.
        """
        if para_evol is None:
            if hasattr(self, 'para_evol'):
                para_evol = self.para_evol
            else:
                print("No parameter evolution data available.")
                return
        if func_to_track is None:
            if hasattr(self, 'func_to_track'):
                func_to_track = self.func_to_track
            else:
                print("No track function available.")
                return
            
        burnin_step = int(len(para_evol) * burnin_rate)
        if func_evol is None:
            if hasattr(self, 'func_evol'):
                func_evol = self.func_evol
            else:
                func_evol = [func_to_track(para) for para in para_evol[:burnin_step]]

        if len(self.func_evol_SA) < burnin_step:
            self.para_evol_SA = para_evol[:burnin_step]
            self.func_evol_SA = func_evol[:burnin_step]

        para_evol = np.array(para_evol)
        for i in range(max(burnin_step, len(self.func_evol_SA)), len(para_evol)):
            avg_para = np.mean(para_evol[max(0, i-SA_num+1):i+1], axis=0)
            self.para_evol_SA.append(avg_para)
            self.func_evol_SA.append(func_to_track(avg_para))
        self.latest_results['para_evol_SA'] = self.para_evol_SA.copy()
        self.latest_results['func_evol_SA'] = self.func_evol_SA.copy()
        if store_results_in_self:
            self.results_list[self._results_index] = self.latest_results.copy()
        return self.latest_results.copy()
    ###############
        
    def calculate_BMA(self, func_evol=None, BMA_num=10, burnin_rate=0.9, store_results_in_self=True):
        """
        Calculates the Bayesian model average of `func_evol` for each step, taking into account the last `BMA_num` function values.
        If `BMA_num` is larger than the available function values, the function uses as many as available for the average.
        It then appends the result to `self.func_evol_BMA` and to the 'func_evol_BMA' element of `self.latest_results`.

        Parameters:
        - func_evol (list): The evolution of a function. If None, `self.func_evol` will be used. If that's also None, the function ends with a message.
        - BMA_num (int): The number of last function values to consider for the average. Default is 10.
        - burnin_rate (float): Fraction of total iterations after which the suffix averaging begins. Default is 0.9.
        - store_results_in_self (bool): Whether to store the results in `self.results_list[self._results_index]`. Default is True.
        """
        if func_evol is None:
            if hasattr(self, 'func_evol'):
                func_evol = self.func_evol
            else:
                print("No function evolution data available.")
                return

        burnin_step = int(len(func_evol) * burnin_rate)
        if len(self.func_evol_BMA) < burnin_step:
            self.func_evol_BMA = func_evol[len(self.func_evol_BMA):burnin_step]

        recent_func_vals = [func for func in func_evol[max(0, len(self.func_evol_BMA)-BMA_num):len(self.func_evol_BMA)] if func is not None]
        
        for i in range(max(burnin_step, len(self.func_evol_BMA)), len(func_evol)):
            if func_evol[i] is not None:
                recent_func_vals.append(func_evol[i])
                if len(recent_func_vals) > BMA_num:
                    recent_func_vals.pop(0)
                first_element = recent_func_vals[0]
                # If the elements of recent_func_vals are lists
                if isinstance(first_element, list):
                    avg_func = [np.mean([val[j] for val in recent_func_vals]) for j in range(len(first_element))]
                # If the elements of recent_func_vals are dictionaries
                elif isinstance(first_element, dict):
                    keys = first_element.keys()
                    avg_func = {}
                    for key in keys:
                        avg_func[key] = np.mean([val[key] for val in recent_func_vals])
                else:
                    avg_func = np.mean(recent_func_vals)
                self.func_evol_BMA.append(avg_func)
            else:
                self.func_evol_BMA.append(None)
        self.latest_results['func_evol_BMA'] = self.func_evol_BMA.copy()
        if store_results_in_self:
            self.results_list[self._results_index] = self.latest_results.copy()
        return self.latest_results.copy()