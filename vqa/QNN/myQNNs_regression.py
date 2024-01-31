import os

import pandas as pd
import itertools
import copy
import seaborn as sns

from vqa.VQA_optimizers import qml_optimizers
from vqa.VQA_optimizers.base import init_logging

import seaborn as sns

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


def min_length(*arrays):
    return min(map(len, arrays))

import matplotlib.ticker as ptick

from multiprocessing import cpu_count

import dill
import hashlib
import uuid
from datetime import datetime



def evaluate_expression(expression, variables):
    while any(f"${var_name}$" in expression for var_name in variables.keys()):
        for var_name, var_value in variables.items():
            wrapped_var_name = f"${var_name}$"  # 変数名を$で囲む
            if wrapped_var_name not in expression:
                continue

            # 変数の値が文字列であり、さらに$$で囲んだ変数名を含む場合には再帰的に処理
            if isinstance(var_value, str) and any(f"${inner_var}$" in var_value for inner_var in variables.keys()):
                var_value = evaluate_expression(var_value, variables)

            if isinstance(var_value, np.ndarray):
                # NumPy 配列を文字列化
                array_str = np.array2string(var_value, separator=',', threshold=np.inf)
                array_str = array_str.replace('\n', '').replace(' ', '')
                expression = expression.replace(wrapped_var_name, f'np.array({array_str})')
            else:
                expression = expression.replace(wrapped_var_name, str(var_value))
    return eval(expression)


def make_hashable(params):
    hashable_items = []
    for k, v in params.items():
        try:
            hash(v)  # これが成功すれば、vはハッシュ可能
            hashable_items.append((k, v))
        except TypeError:  # ハッシュ不可能な値に対しては、strに変換
            hashable_items.append((k, str(v)))
    return frozenset(hashable_items)

class HyperParameterManager:
    
    def __init__(self, base_dir="hp_results", base_dir_metadata='hp_metadata', label=None, load_metadata=False, metadata_file_name=None):
        '''
        add_results(hpara_dict, results)で、hparaと対応するresultsを、
        self.dataという辞書に、hpara_dict.items()のfrozensetをキーとして、resultsをvalueとして保持して、
        ファイルに保存する。ファイルは、各hparaのresultsごとにadd_resultsするたびに保存し、
        ファイル名も、self.file_data に、同じキーで保持される。
        '''
        self.data = {}
        self.file_data = {}
        self._base_dir = base_dir
        self._base_dir_metadata = base_dir_metadata
        self.label = label or str(uuid.uuid4())
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if load_metadata:
            if metadata_file_name:
                file_to_load = f"{self._base_dir_metadata}/{metadata_file_name}"
            else:
                # Get the latest file from base_dir_metadata
                file_to_load = self._get_latest_metadata_file()
                print(f"Loading from latest metadata file: {file_to_load}")

            self.load_metadata(file_to_load)
        
    def _generate_id(self, params):
        param_str = str(params)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _generate_metadata_filename(self):
        date_str = datetime.now().strftime('%Y%m%d')
        return f"{self._base_dir_metadata}/{date_str}_{self.label}_metadata.pkl"
    
    def _get_latest_metadata_file(self):
        files = [f for f in os.listdir(self._base_dir_metadata) if os.path.isfile(os.path.join(self._base_dir_metadata, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self._base_dir_metadata, x)), reverse=True)
        if files:
            return os.path.join(self._base_dir_metadata, files[0])
        else:
            raise ValueError("No metadata files found!")
    
    def add_result(self, params, results, label=None):
        param_id = self._generate_id(params)
        date_str = datetime.now().strftime('%Y%m%d')
        use_label = label or self.label
        filename = f"{self._base_dir}/{date_str}_{use_label}_{param_id}.pkl"
        
        frozen_params = make_hashable(params)
        self.data[frozen_params] = results
        self.file_data[frozen_params] = filename

        with open(filename, 'wb') as f:
            dill.dump(results, f)

    def get_result(self, **params):
        '''
        ハイパーパラメータたちの辞書があるときは、**つけてアンパックして渡すこと
        '''
        frozen_params = make_hashable(params)
        result = self.data.get(frozen_params)

        if not result:
            # If not present in memory, try loading from file
            filename = self.file_data.get(frozen_params)
            if filename:
                try:
                    with open(filename, 'rb') as f:
                        result = dill.load(f)
                        self.data[frozen_params] = result
                        print(f"Loaded result from file: {filename}")
                except:
                    return "Error occurred while loading the result from file."
            else:
                return "No matching result or file found."

        return result

    def search_results(self, **params):
        search_set = make_hashable(params)
        matched_data = {}

        for param_key, result in self.data.items():
            if search_set.issubset(param_key):
                matched_data[frozenset(dict(param_key))] = result

        # If matched data is empty, try loading from files
        if not matched_data:
            for param_key, filename in self.file_data.items():
                if search_set.issubset(param_key):
                    try:
                        with open(filename, 'rb') as f:
                            result = dill.load(f)
                            self.data[param_key] = result
                            matched_data[frozenset(dict(param_key))] = result
                            print(f"Loaded result from file: {filename}")
                    except:
                        continue

        if not matched_data:
            return "No matching results or files found."
        
        self.searched_data = matched_data
        return matched_data

    def save_metadata(self, filename=None):
        if not filename:
            filename = self._generate_metadata_filename()
        with open(filename, 'wb') as f:
            dill.dump(self.file_data, f)
    
    def load_metadata(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.file_data = dill.load(f)
        except:
            print("Error occurred while loading metadata.")

    def search_files(self, **params):
        '''
        ファイルデータをロードしてから使う
        Usage:
        # manager = HyperParameterManager(load_metadata=True)
        # matched_files = manager.search_files(some_param=value, another_param=another_value)
        # for f in matched_files:
        #     print(f)
        '''
        search_set = make_hashable(params)
        matched_files = []

        for param_key, filename in self.file_data.items():
            if search_set.issubset(param_key):
                matched_files.append(filename)
        
        print(matched_files)
        return matched_files



def label2one_hot(label, num_classes):
    one_hot_list = np.zeros(num_classes, dtype=float)
    one_hot_list[label] = 1.
    return one_hot_list

def sum_cross_products(X):
    total_sum = np.sum(X)
    sums_without_i = total_sum - X
    cross_products = sums_without_i * X
    return np.sum(cross_products)

def split_array_equally(arr):
    # 配列の長さを取得
    length = len(arr)
    
    # 分割するインデックスを計算
    mid = length // 2

    # 配列を分割
    part1 = arr[:mid]
    part2 = arr[mid:2*mid]  # 2*mid で奇数の場合の最後の要素を除外
    
    return part1, part2

def add_partial_kwargs(manual_partial_kwargs, default_tuple):
    """
    manual_partial_kwargsの形式に応じて適切な形に変換し、default_tupleを追加する。（すでにある場合は上書き）
    """
    # manual_partial_kwargsがタプルの場合、辞書の形式に変換
    if isinstance(manual_partial_kwargs, tuple):
        manual_partial_kwargs = {manual_partial_kwargs[0]: manual_partial_kwargs[1]}

    # manual_partial_kwargsがタプルのリストの場合、辞書の形式に変換
    elif isinstance(manual_partial_kwargs, list) and all(isinstance(item, tuple) for item in manual_partial_kwargs):
        manual_partial_kwargs = dict(manual_partial_kwargs)

    # default_tupleを追加
    manual_partial_kwargs[default_tuple[0]] = default_tuple[1]

    return manual_partial_kwargs

class myQNN_regression:
    '''
    learning_circuit : skqulacsのlearning_circuit クラスを入れる。
    jebri et al などで使われたタスク(hardware efficient dataset の regression)を想定
    obsで与えられたオブザーバブルの期待値*params[-1]の値で yの値を回帰。
    normalization parameter w がtrainable parameterとして追加されている
    optimizer の対象となる target_obj として与えられ, clientとして使われる
    最適化に使う勾配計算などの提供や、内部カウンターなどの連携のみに使う。（並列化するときは、別のインスタンスを処理ごとにつくるなど同期に注意）
    最適化の結果の管理は別で。
    '''
    def __init__(self, n_qubits, learning_circuit, 
                 x_train=None, y_train=None, x_test=None, y_test=None, min_approx_shots=-1, setting_dict=None, id_string="QNN", file_id=None, obs=None,                 
                shot_time=1., per_circuit_overhead=0., communication_overhead=0., max_sub_nqubits_sampling_aft_partial_tr=6):
        '''
        min_approx_shots: 何ショット以上はCLTで近似するか。-1とすると近似なし。
        setting_dict: optimizerで使う gradや iEval などを指定する
        max_sub_nqubits_sampling_aft_partial_trはobservableのサンプリングをするときに、ターゲットが何qubitまでpartial traceとってからサンプリングするか
        （計算速度に関わる。大体実験したら、少数では圧倒的にpartial traceだが、全qubit数が大きい場合、半分ぐらいから微妙になってくるので、
        ## 9-qubitまでは、常にpartial trace
        ## 10-qubit では k=7 からdirect
        ## 12-qubitぐらい以上の k=5あたりからはdirect samplingでmodとるほうがよさげ。partial traceをとる時間と、部分系のdensity matrixの次元が大きくなる影響と思われる。）
        ただし今は常にpartial traceとっていて、切り替えは未実装なので今は気にしない。（observableとして Z0を前提としたものしか作ってない）
        '''
        global generate_labels_flag
        generate_labels_flag = False        
        self.circuit = learning_circuit
        self.n_c_params = len(self.circuit.get_parameters())
        self.n_params = self.n_c_params + 1 #re-normalization coeffのパラメータ１つ分多い
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
        if obs is None:
            obs = Observable(n_qubits)
            obs.add_operator(1., 'Z 0')
            self._tr_out_ind = [i for i in range(1, self.n_qubits)]
        self.obs_json=obs.to_json() #拡張するとき用。今は固定したものしか使わないので、これは使わない。（実行時間は読み込みのほうが遅い）
        ###これらは、一般のオブザバブルで推定する場合に使う。（とりあえず未実装で、Z0だけ作った）
        self.target_indices_list = []
        for i in range(obs.get_term_count()):
            term = obs.get_term(i)
            self.target_indices_list.append(term.get_index_list())
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
            self.x_train = np.array(x_train)
            self.y_train = np.array(y_train)
            self.num_train_data = len(x_train)
            self._uniform_wt = np.full(len(self.x_train), 1.) #計算されるショット数は、各データ点への割当が直接出るようにしたので、１
            self._uniform_wt_dist = np.full(len(self.x_train), 1.) / len(self.x_train)
            self._max_abs_y_train = np.max(np.abs(self.y_train))
            #self.y_train_1hot = np.array([label2one_hot(y_train[i], num_classes) for i in range(len(y_train))])
        if (not (x_test is None)) and (not (y_test is None)):
            self.x_test = np.array(x_test)
            self.y_test = np.array(y_test)
        

    def remove_data(self):
        # x_train が存在すれば、削除
        if hasattr(self, 'x_train'):
            del self.x_train
        
        # y_train が存在すれば、削除
        if hasattr(self, 'y_train'):
            del self.y_train
            
        # num_train_data が存在すれば、削除
        if hasattr(self, 'num_train_data'):
            del self.num_train_data
            
        # y_train_1hot（注釈からはコメントアウトされているので、ここではオプションとして追加）が存在すれば、削除
        # if hasattr(self, 'y_train_1hot'):
        #     del self.y_train_1hot
            
        # x_test が存在すれば、削除
        if hasattr(self, 'x_test'):
            del self.x_test
            
        # y_test が存在すれば、削除
        if hasattr(self, 'y_test'):
            del self.y_test

    def get_data(self):
        # x_train の存在を確認して返す。存在しない場合は空のリストを返す。
        x_train = getattr(self, 'x_train', [])

        # y_train の存在を確認して返す。存在しない場合は空のリストを返す。
        y_train = getattr(self, 'y_train', [])

        # x_test の存在を確認して返す。存在しない場合は空のリストを返す。
        x_test = getattr(self, 'x_test', [])

        # y_test の存在を確認して返す。存在しない場合は空のリストを返す。
        y_test = getattr(self, 'y_test', [])

        return x_train, y_train, x_test, y_test

    ## file save utilities
    def save_to_file(self, results=None, save_self_obj=True, directory=None):
        '''
        一回の最適化が終わったときに、resultsまたは自身のオブジェクトを保存するためのメソッド
        '''
        # Combine the optimizer name, file id, and file name into the filename
        try:
            if directory:
                # Make sure the directory exists. If not, create it.
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = f"{directory}/{self.id_string}_model_{self._optimizer_name}_{self.file_id}.pkl"
            else:
                file_name = f"{self.id_string}_model_{self._optimizer_name}_{self.file_id}.pkl"
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
                #dill なら、循環参照していてもいいらしい
                #if hasattr(self_copy, 'optimizer'):
                #    if hasattr(self_copy.optimizer, 'target_obj'):
                #        delattr(self_copy.optimizer, 'target_obj')
                # Removing reference to target_obj object to avoid dill error
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
            search_string = f"{self.id_string}"  # Default search string
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

        This method attempts to load data from a dill file located at the
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

    def set_overhead_time(self, shot_time=None, per_circuit_overhead=None, communication_overhead=None):
        if shot_time is not None:
            self.shot_time = shot_time
        if per_circuit_overhead is not None:
            self.per_circuit_overhead = per_circuit_overhead
        if communication_overhead is not None:
            self.communication_overhead = communication_overhead


    def set_methods_for_optimizer(self, setting_dict):
        '''
        最適化に使うcostやgradなどの計算メソッドを設定する。
        
        setting_dict:
            keys: optimizerで使用される項目名,
            values(str): 対応するメソッドの名前
            self.methods_for_optimization 辞書に保持される。
        note: 各メソッドを最適化に使うときにoptionalな引数を指定したいときは、optimizeするときに**kwargsの一部として与える
        '''
        
        for key, method_name in setting_dict.items():
            # メソッドを取得
            method = getattr(self, method_name)  
            
            # 辞書にメソッドをセット
            self.methods_for_optimization[key] = method
            
            # 確認のための出力
            #print(f"Set for {key}: {method_name}")


    def compile(self, optimizer='SGD', loss='mean_squared_error', hpara=None, init_params=None, min_approx_shots=None,
                x_train=None, y_train=None, x_test=None, y_test=None,
                shot_time=None, per_circuit_overhead=None, communication_overhead=None,
                init_spec_dict=None, manual_methods_setting_dict=None):
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
        self._optimizer_name = optimizer
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
                    'func_to_track': 'MSE_both_exact_loss_to_track',
                    'loss': 'MSE_loss_eval',
                    'grad': 'MSE_grad_eval',
                    'iEvaluate': 'MSE_iEval',
                    'iEvaluate_iRandom': 'MSE_iEval_iRandom'
                }
                self.set_methods_for_optimizer(methods_setting_dict)
        obj = self
        #for attr_name in dir(obj):
        #    if attr_name.startswith("__"):  # 組み込み属性（magic methodなど）をスキップ
        #        continue
        #    attr_value = getattr(obj, attr_name)
        #    #logging.debug(f"{attr_name}: {type(attr_value)}")
        #    print(f"{attr_name}: {type(attr_value)}")
        for attr_name, attr_value in vars(obj).items():
            try:
                dill.dumps(attr_value)
            except Exception as e:
                print(f"Cannot pickle the attribute {attr_name}: {e}")
        init_spec_dict['target_obj'] = copy.deepcopy(self)
        self.optimizer = optimizer_class(hpara=hpara, init_spec_dict=init_spec_dict)
        self._optimized = False

    def fit(self, init_params=None, options=None, hpara=None, init_spec_dict=None, Return_results=True, store_results_in_self=False,
            auto_save=False, save_self_obj=True, save_directory='single_optimization', process_id=0, **kwargs):
        '''
        Do compile before fit.
        最適化を実行。
        compileすることでリセットされ、そうでない場合は呼び出す度につづきから最適化をする。（最適化終了条件は変更して与えないとアップデートされない）

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
            'take_SA': False, #suffix averageをとった結果もresultsに含めるか
            'take_BMA': False, #BMAをとった結果もresultsに含めるか
            'shot_time': 1., #overhead時間をあとから指定することも可能。
            'per_circuit_overhead': 0.,
            'communication_overhead': 0.
        }
        '''
        if not hasattr(self, 'optimizer'):
            print('Optimizer has not been specified')
            return
        if options is None:
            options = {}
        options['process_id'] = process_id
        if not self._optimized:
            #はじめから
            #print('First optimization')
            if init_spec_dict is None:
                init_spec_dict = {}
            if init_params is not None:
                init_spec_dict['init_params'] = init_params
                self.model_params = init_params
            init_spec_dict['target_obj'] = self
            results = self.optimizer.new_optimize(init_spec_dict=init_spec_dict, hpara=hpara, options=options, **kwargs)
            self.model_params = results['para_evol'][-1]
            if self._optimizer_name not in self.results_dict:
                self.results_dict[self._optimizer_name] = []
            if store_results_in_self:
                self.results_dict[self._optimizer_name].append(results)
            else:
                Return_results = True
            self.elapsed_time_for_optimization = self.optimizer.elapsed_time
            self._optimized = True
        else:
            #つづきから
            #print('Continue optimization (init_params, init_spec_dict, hpara are ignored)')
            results = self.optimizer.continue_optimize(options_to_update=options, kwargs_to_update=kwargs)
            self.model_params = results['para_evol'][-1]
            if store_results_in_self:
                self.results_dict[self._optimizer_name][-1] = results #続きなので、最後の要素を上書きする。
            else:
                Return_results = True
            self.elapsed_time_for_optimization = self.optimizer.elapsed_time
            self._optimized = True
        if auto_save:
            self.save_to_file(results=results, save_self_obj=save_self_obj, directory=save_directory)
        if Return_results:
            return results
        else:
            return
            
    def _eval_state_exact_expect(self, state):
        '''
        exact expectation of the obs with the given state
        '''
        #caller = inspect.currentframe().f_back
        ##logging.debug(f"Called from function {caller.f_code.co_name} in {caller.f_code.co_filename}:{caller.f_lineno}")
        ##logging.debug('just before using self.observable')
        #obs = qulacs.observable.from_json(self.obs_json)
        obs = Observable(self.n_qubits)
        obs.add_operator(1., 'Z 0')
        #tmp_state = state.copy()
        #traced_state = partial_trace(tmp_state, self._tr_out_ind)
        expect = obs.get_expectation_value(state)
        return (expect).real
    
    def _eval_state_exact_expect_var_singlePauli(self, state):
        '''
        exact expectation and its variance of the observable (single Pauli) with the given state
        期待値だけサンプリングをCLT近似するときに使う
        '''
        exp_val = self._eval_state_exact_expect(state)
        var = 1. - exp_val**2
        return exp_val, var
     
    def _sample_state_Z0(self, state, n_shots, Return_var=False, Return_outcomes=False):
        '''
        given state に対して、Z0のサンプリングをする。基本的に期待値を返す。
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
            
    def _sample_outcomes_Z0_single_data(self, x, n_shots):
        '''
        self.circuitのパラメータをセットした前提。
        一つのデータで、ガチサンプリングして出たoutcomesを返す。
        '''
        state = self.circuit.run(x)
        self.circuit_counter[0] += 1
        return self._sample_state_Z0(state, n_shots, Return_outcomes=True)
    
    def _sample_outcomes_sum_Z0_all_data(self, params, x_eval, n_s_list):
        '''
        n_s_list[i] i-th dataに使うショット数のリスト（0でないものだけを取り出したもの推奨）
        x_eval のdata点のインデックスは、n_s_listのindexと同期している必要がある。
        n_tot: 分配元の、total number of shots. n_s_listが2倍されているなら、n_totも２倍。
        order = 1 なら、各データでの expvalたちの array
        orderがそれ以上なら、EX_list[j] に、各データ点での、期待値 j+1乗のarrayが入った２重リスト
        expvalとそのpower は、重みをかけたものを返す。
        '''
        self.circuit.update_parameters(params[:self.n_c_params])
        outcomes_data_list = [] #各要素に、i-th data点でのoutcomesが入る。要素ごとに長さが違う。
        outsum_data_list = np.zeros(len(x_eval))
        for i in range(len(x_eval)):
            outcomes = self._sample_outcomes_Z0_single_data(x_eval[i], n_s_list[i])
            outcomes_data_list.append(outcomes)
            outsum_data_list[i] = np.sum(outcomes)
        return outcomes_data_list, outsum_data_list

    # multinomial で random allocationするときは単純に期待値とれないのでボツ #
    # def _sample_outcomes_Z0_all_data(self, params, x_eval, n_s_list, n_tot,,order=2):
    #     '''
    #     n_s_list[i] i-th dataに使うショット数のリスト（0でないものだけを取り出したもの推奨）
    #     x_eval のdata点のインデックスは、n_s_listのindexと同期している必要がある。
    #     n_tot: 分配元の、total number of shots. n_s_listが2倍されているなら、n_totも２倍。
    #     order = 1 なら、各データでの expvalたちの array
    #     orderがそれ以上なら、EX_list[j] に、各データ点での、期待値 j+1乗のarrayが入った２重リスト
    #     expvalとそのpower は、重みをかけたものを返す。
    #     '''
    #     self.circuit.update_parameters(params[:self.n_c_params])
    #     outcomes_data_list = [] #各要素に、i-th data点でのoutcomesが入る。要素ごとに長さが違う。
    #     if order == 1:
    #         EX_list = np.zeros(len(x_eval))
    #     else:
    #         EX_list = [np.zeros(len(x_eval)) for i in range(order)]
    #     for i in range(len(x_eval)):
    #         outcomes = self._sample_outcomes_Z0_single_data(x_eval[i], n_s_list[i])
    #         outcomes_data_list.append(outcomes)
    #         expval = np.sum(outcomes) / n_tot
    #         if order == 1:
    #             EX_list[i] = expval
    #         else:
    #             EXpow = self._unbiased_powerE_Pauli(expval, n_s_list[i], order=order, unbiased=True)
    #             for j in range(order):
    #                 EX_list[j][i] = EXpow[j]
    #     return EX_list, outcomes_data_list
    
    #とりあえず、SantaやCANSの分散の推定で使用する、積を一つのサンプルとした不偏分散は、CLT未実装。
    #CLT近似をするときには、outcomeのリストを返すことができない。べき乗の不偏推定量を出したいときには、
    #統計量で表す
    #projectionなので、２乗が等しいため、sample varianceは、\sum_i z_i だけでかけるので、covはいらない   
    def _single_state_hybrid_sampler(self, state, n_shots, min_approx_shots, Return_var=False):
        '''
        observableが１つのPauliであること前提で期待値と分散の推定を行う。
        stateを与えてサンプリングをする。
        単一のデータでの値の計算（data点固定での期待値と分散）
        '''
        self.call_counter[0] += 1 #single stateへの evaluationでとりあえず
        if n_shots < min_approx_shots or min_approx_shots < 0:
            return self._sample_state_Z0(state, n_shots, Return_var=Return_var)
        else:
            #CLTで近似
            #exact_label = self.exact_pred(state)
            exact_label, exact_var = self._eval_state_exact_expect_var_singlePauli(state) #期待値と分散を同時に返す
            exp_val_est = exact_label + randn()*np.sqrt(exact_var / n_shots)
            self.shots_counter[0] += n_shots
            if Return_var:
                sample_var = float(n_shots) / (float(n_shots) - 1.) *(1. - exp_val_est**2)
                return exp_val_est, sample_var
            else:
                return exp_val_est
            
    def _unbiased_powerE_Pauli(self, sample_mean, n_samples, order=4, unbiased=False):
        '''
        sample_meanとサンプル数n_samplesを入れて、2からorderまでの期待値べき乗の不偏推定量を返す。
        ただし、O^2=identity なる物理量の期待値限定
        ここでは、ショット数がdeterministicな場合だけ考える（QEMに適用するには修正が必要）
        '''
        n = n_samples
        M_n =sample_mean
        if unbiased:
            if order==1:
                EO_list = [M_n]
            if order==2:
                EO2 = (n/(n-1.))*M_n**2 - 1/(n-1.)
                #print(EO2)
                #EO2 = max(EO2, 1e-10)
                EO_list = [M_n, EO2]
            elif order==3:
                EO2 = (n/(n-1.))*M_n**2 - 1/(n-1.)
                #print(EO2)
                #EO2 = max(EO2, 1e-10)
                EO3 = (n/(n-1.))*(n/(n-2.))*M_n**3 - (3.*n-2)/(n-1.)/(n-2.) * M_n
                EO_list = [M_n, EO2, EO3]
            elif order>=4:
                EO2 = (n/(n-1.))*M_n**2 - 1/(n-1.)
                #print(EO2)
                #EO2 = max(EO2, 1e-10)
                EO3 = (n/(n-1.))*(n/(n-2.))*M_n**3 - (3.*n-2)/(n-1.)/(n-2.) * M_n
                EO4 = (n/(n-1.))*(n/(n-2.))*(n/(n-3.))*M_n**4 - ((n/(n-1))*((6.*n-8.)/(n-2.))*M_n**2 - 3./(n-1.))/(n-3.)
                #EO4 = max(EO4, 1e-10) あとからクリップしたほうがいい
                EO_list = [M_n, EO2, EO3, EO4]
        else:
            if order==1:
                EO_list = [M_n]
            if order==2:
                EO2 = M_n**2
                EO_list = [M_n, EO2]
            elif order==3:
                EO2 = M_n**2
                EO3 = M_n**3
                EO_list = [M_n, EO2, EO3]
            elif order>=4:
                EO2 = M_n**2
                EO3 = M_n**3
                EO4 = M_n**4
                EO_list = [M_n, EO2, EO3, EO4]
        return EO_list
    
    def _single_state_hybrid_sampler_power(self, state, n_shots, min_approx_shots, order=4, Return_var=False, unbiased=False):
        '''
        stateを入れて、hybrid samplingをするやつ。powerの推定量を返す。
        とりあえず、ショット数はdeterministic
        '''
        self.call_counter[0] += 1
        #for debug
        if min_approx_shots == -10:
            #debugのため、exactな値を返す。
            #self.shots_counter[0] += n_shots
            exact_label, exact_var = self._eval_state_exact_expect_var_singlePauli(state)
            EO_list = np.array([exact_label**i for i in range(1, order + 1)])
            if Return_var:                
                return EO_list, exact_var
            else:
                return EO_list
        if n_shots < min_approx_shots or min_approx_shots < 0:
            if Return_var:
                exp_val, sample_var = self._sample_state_Z0(state, n_shots, Return_var=Return_var)
                EO_list = self._unbiased_powerE_Pauli(exp_val, n_shots, order=order, unbiased=unbiased)
                return EO_list, sample_var
            else:
                exp_val = self._sample_state_Z0(state, n_shots, Return_var=Return_var)
                EO_list = self._unbiased_powerE_Pauli(exp_val, n_shots, order=order, unbiased=unbiased)
                return EO_list
        else:
            #CLTで近似
            #exact_label = self.exact_pred(state)
            exact_label, exact_var = self._eval_state_exact_expect_var_singlePauli(state) #同時に返す
            #print('label', exact_label)
            #print('var', exact_var)
            exp_val_est = exact_label + randn()*np.sqrt(exact_var / n_shots)
            self.shots_counter[0] += n_shots #測定した場所
            EO_list = self._unbiased_powerE_Pauli(exp_val_est, n_shots, order=order,unbiased=unbiased)
            if Return_var:
                sample_var = float(n_shots) / (float(n_shots) - 1.) * (1. - exp_val_est**2)
                return EO_list, sample_var
            else:
                return EO_list
    
    def _exact_eval_obs_single_data(self, x):
        '''
        self.circuitのパラメータはすでに設定されているとして、run(x)してobsのexactな期待値を返す。
        ただし、これ自体がpredictの値とは限らないことに注意（スケールパラメータをかけたりする）
        サブルーチンとして使う
        '''
        state = self.circuit.run(x) #self.circuitで、パラメータをセットしてデータxの状態に作用させた状態を作る
        return self._eval_state_exact_expect(state)
    
    def exact_predict(self, x_list, params=None):
        '''
        self.circuitにパラメータをセットし、x_listに与えたデータたちの exactな predictionのリスト pred_listを返す。
        '''
        if params is None:
            params = self.model_params
        self.circuit.update_parameters(params[:self.n_c_params])
        pred_list = params[-1] * np.array([self._exact_eval_obs_single_data(x) for x in x_list])
        return pred_list
    
    def exact_MSE(self, params, which="test", ret_both=False, log_evol=False):
        #logging.debug('exact_MSE just called')
        if ret_both:
            which="test"
        if which=="train":
            y_pred = self.exact_predict(self.x_train, params).astype(float)
            #logging.debug('just after exact_predict')
            y_train = self.y_train
            MSE = np.sum(np.square(y_pred - y_train)) / len(y_train)
            if log_evol:
                self.exact_cost_evol.append(self.exact_MSE(params, which="test"))
            return MSE
        elif which=="test":
            #logging.debug('before test data processing')
            y_pred = self.exact_predict(self.x_test, params).astype(float)
            y_test = self.y_test
            MSE = np.sum(np.square(y_pred - y_test)) / len(y_test)
            if log_evol:
                self.exact_cost_evol.append(MSE)
            if ret_both:
                #logging.debug('just before call exact_MSE for train')
                MSE_train = self.exact_MSE(params, which='train', ret_both=False)
                return {'test': MSE, 'train': MSE_train}
            else:
                return MSE

    #最適化で得られたパラメータで、別のdataに対してBMA計算したいときなどに使う        
    def exact_BMA_predict(self, x_list, params_list, num_avg):
        '''
        params_list は、複数ステップのパラメータのアンサンブル
        num_avg個でmodel average
        '''
        num_avg = min(len(params_list), num_avg)
        para_avg_list = params_list[-num_avg:]
        pred_list_list = np.array([self.exact_predict(x_list, params) for params in para_avg_list])
        return np.mean(pred_list_list, axis=0)
    
    def exact_BMA_MSE(self, params_list, num_avg, which='test', ret_both=False):
        if ret_both:
            which="test"
        if which=="train":
            y_pred = self.exact_BMA_predict(self.x_train, params_list, num_avg).astype(float)
            y_train = self.y_train
            MSE = np.sum(np.square(y_pred - y_train)) / len(y_train)
            return MSE
        elif which=="test":
            y_pred = self.exact_BMA_predict(self.x_test, params_list, num_avg).astype(float)
            y_test = self.y_test
            MSE = np.sum(np.square(y_pred - y_test)) / len(y_test)
            if ret_both:
                MSE_train = self.exact_BMA_MSE(params_list, num_avg, which='train', ret_both=False)
                return {'test': MSE, 'train': MSE_train}
            else:
                return MSE
            
    def MSE_exact_train_loss(self, params):
        '''
        MSE
        exactな train lossだけlog_evolにとりあえずトラックする用
        '''
        return self.exact_MSE(params, which="train", log_evol=True)
    
    def MSE_both_exact_loss_to_track(self, params):
        '''
        MSE
        exactな trainとtest loss両方をtrack するとき用
        '''
        #logging.debug('MSE_both_exact just called')
        return self.exact_MSE(params, which="test", ret_both=True, log_evol=False)
            
    def MSE_both_noisy_loss_to_track(self, params):
        '''
        optimizationの途中で呼び出して、current valueを計算する用
        '''
        y_test = self.y_test
        y_train = self.cache_y_train_eval
        n_s_list = np.full(len(y_test), self.n_shots_test)
        test_pred = self._sample_labels(params, self.x_test, n_s_list, min_approx=self.min_approx_shots, Return_var=False)
        test_MSE = np.sum(np.square(test_pred - y_test)) / len(y_test)
        if hasattr(self, 'cache_train_pred'):
            train_MSE = np.sum(np.square(y_train - self.cache_train_pred)) / len(y_train)
        else:
            train_pred = self._sample_labels(params, self.x_train, n_s_list, min_approx=self.min_approx_shots, Return_var=False)
            train_MSE = np.sum(np.square(y_train - train_pred)) / len(y_train)
        return {'test': test_MSE, 'train': train_MSE}
        
    def reset_exact_cost_evol(self):
        self.exact_cost_evol = []
    
    def _sample_obs_single_data_no_power(self, x, n_shots, option_min_approx=None, Return_var=False):
        '''
        self.circuitにparameterがセットされた前提で、
        single dataで、observableのサンプル平均、サンプル不偏分散（Return_var=Trueのとき）を返す。
        powerつかわないやつ
        '''
        min_approx = option_min_approx
        state = self.circuit.run(x)
        self.circuit_counter[0] += 1 #circuitをrunした場所
        return self._single_state_hybrid_sampler(state, n_shots, min_approx, Return_var=Return_var)
    
    def _sample_obs_single_data(self, x, n_shots, option_min_approx=None, order=4, Return_var=False, unbiased=False):
        '''
        self.circuitにparameterがセットされた前提で、
        single dataで、observableのサンプル平均、サンプル不偏分散（Return_var=Trueのとき）を返す。
        '''
        min_approx = option_min_approx
        state = self.circuit.run(x)
        self.circuit_counter[0] += 1
        return self._single_state_hybrid_sampler_power(state, n_shots, min_approx, order=order, Return_var=Return_var, unbiased=unbiased)
    
    #データごとのpredictのリストと分散
    def sample_predict(self, x_eval, n_s_list, params=None, min_approx=None, Return_var=False):
        '''
        データ列 x_eval (list)の各データすべてについて n_s_list の要素のショット数を使って、observableの測定をし、
        params[-1]でスケーリングして最終的なpredictのリストを返す。
        Return_var=Trueならその分散も返す
        predict、分散それぞれの配列のshapeは、(len(x_eval),)
        '''
        if min_approx is None:
            min_approx = self.min_approx_shots
        if params is None:
            params = self.model_params
        self.circuit.update_parameters(params[:self.n_c_params])
        exp_vals = np.zeros(len(x_eval))
        w = params[-1] #re-normalization coeff
        if Return_var:
            var_lists = np.zeros(len(x_eval))
            for i in range(len(x_eval)):
                exp_p, var_p = self._sample_obs_single_data_no_power(x_eval[i], n_s_list[i], option_min_approx=min_approx, Return_var=Return_var)
                exp_vals[i] = exp_p
                var_lists[i] = var_p
            return w * exp_vals, var_lists * w**2
        else:
            for i in range(len(x_eval)):
                exp_p = self._sample_obs_single_data_no_power(x_eval[i], n_s_list[i], option_min_approx=min_approx, Return_var=Return_var)
                exp_vals[i] = exp_p
            return exp_vals * w

    #データごとのobservableの期待値とべき乗のリストと分散    
    def _sample_obs_data_by_data(self, params, x_eval, n_s_list, min_approx, order=4, Return_var=False, unbiased=False):
        '''
        データ列 x_eval (list)の各データすべてについて n_s_list の要素のショット数を使って、observableの測定をした期待値とべき乗を、
        スケーリングなしで返す。勾配計算のためにつかう。
        Return_var=Trueならその分散も返す
        期待値の配列exp_powers_listのshapeは、(order, len(x_eval))
        つまり、exp_powers_list[j]に、期待値のj+1乗の推定量の各データでの値のリストが入っている。
        分散の配列のshapeは、(len(x_eval),)
        predict するときには、params[-1]をスケーリングとしてかけることに注意。
        '''
        self.circuit.update_parameters(params[:self.n_c_params])
        exp_powers_list = [np.zeros(len(x_eval)) for i in range(order)]
        if Return_var:
            var_lists = np.zeros(len(x_eval))
            for i in range(len(x_eval)):
                exp_p, var_p = self._sample_obs_single_data(x_eval[i], n_s_list[i], option_min_approx=min_approx, order=order, Return_var=Return_var, unbiased=unbiased)
                for j in range(order):
                    exp_powers_list[j][i] = exp_p[j]
                var_lists[i] = var_p
            return exp_powers_list, var_lists
        else:
            for i in range(len(x_eval)):
                exp_p = self._sample_obs_single_data(x_eval[i], n_s_list[i], option_min_approx=min_approx, order=order, Return_var=Return_var, unbiased=unbiased)
                for j in range(order):
                    exp_powers_list[j][i] = exp_p[j]
            return exp_powers_list
                
    #### MSEの勾配の分散(のupper boundの1/n係数)の計算メソッド ###
    # サンプル平均で推定しようとしたやつのミニバッチ版 -> 解析的な式を展開して各項の推定量から計算したほうがよさそうなのでボツ
    ## 分散の推定のupper boundとして、\sum_k r_{0,k,x_j}*r_{i,k,x_j}/n をサンプル平均として使った場合の推定量を返す
    ## CLT使うのはややこしいのでとりあえず未実装。
    ## shiftとらないパラメータでの値を使い回すにあたって、outcomesを保存して引数として与えて使う。
    def _direct_sample_MSEgrad_conditional_varUB_single_data(self, outcomes_no_shift, outcomes_PSR=None, Der_of_scale=False):
        '''
        minibatch basedで、data by data に分散の推定量を得たいときに使う。
        Refoqusなどで、各成分独立にサンプリングしてしまう場合はdata点全体で分散を求めるので、全データのoutcomesのリストを与えて
        _sample_derivative_whole_dataを使う。
        self._sample_Z0_single_data で返したoutcomesのリストを渡す。
        Der_of_scale =True は、scaling parameter (params[-1])のderivativeであることを意味し、それ用の計算を、
        Falseなら、それ以外の成分であることを意味し、outcomes_PSRも使ってそれ用の計算をする。
        outcomes_PSRには、PSRをoutcomeごとにとった一つずつが入っているとする（shiftそれぞれの測定回数は同じとする）
        '''
        ## 分散のupper boundとなるfactorすべてを個別に推定して積や和をとるのも考えたが、これはunbiasedにすると正が保証されないし、
        ## 分散も増えそう誤差が大きくなりそう。
        ## ここで実行するように、サンプル数を犠牲にして unbiased estimator にするほうが、unbiasedness を確保できる。サンプル数が犠牲に
        if Der_of_scale:
            # scale parameterの微分
            pass
        else:
            # 回路parameterの微分
            pass

    def _estimate_MSEgrad_mean_condVar_full(self, y_eval, E0pow_data_list, w=1., Scale_para=True, Eppow_data_list=None, Empow_data_list=None):
        '''
        解析式をもとに、個別の推定量から E_data[ V[grad[i, data]|data] ]の不偏推定を計算して返す。
        分散全体を返す。nショット使うときにスケーリングが違う項も全部
        
        （1/n_shots の係数として使うと、実際の勾配推定量の分散のupper boundとなる）
        Scale_para=True なら scale parameterの勾配の分散を計算
        Epow_data_list[i][j] には、期待値i+1乗の推定量の j-th dataのものを入れる。y_evalのj-th indexと連動させる。
        '''
#         logger = init_logging('EV_estimations', raw_id_for_filename=True)
#         logger.debug(f'E0pow: {E0pow_data_list}')
#         logger.debug(f'Eppow: {Eppow_data_list}')
#         logger.debug(f'Eppow: {Empow_data_list}')
#         logger.debug(f'w: {w}')
        EX = np.array(E0pow_data_list[0]) #それぞれ、data点をインデックスとしたnumpy array
        EX_2 = np.array(E0pow_data_list[1])       
        if Scale_para:
            EX_3 = np.array(E0pow_data_list[2])
            EX_4 = np.array(E0pow_data_list[3])
            a = -2 * y_eval
            b = 2 * w            
            est = a**2 + 2*b**2 + 4 * a * b * EX - a**2 * EX_2 - 4 * a * b * EX_3 - 2 * b**2 * EX_4
            Var_X = 1 - np.square(EX)
            est2 = np.square(a + 2*b * EX) * Var_X + 2*b**2 * np.square(Var_X)
#             logger.debug(f'a: {a} \n b: {b}')
#             logger.debug(f'EX: {EX}\n EX2: {EX_2} \n EX3: {EX_3} \n EX4: {EX_4}')
#             logger.debug(f'est: {est}')
#             logger.debug(f'Var_X: {Var_X}')
#             logger.debug(f'est2: {est2}')
#             logger.debug(f'mean est2: {np.mean(est2)}')
            mean_condVar = np.mean(est) #これが分散の 1/mn に比例する項の分子の不偏推定量
            if mean_condVar < 0:
                Var_X = 1 - np.square(EX)
                est = np.square(a + 2*b * EX) * Var_X + 2*b**2 * np.square(Var_X)
                mean_condVar = np.mean(est)
        else:
            EXp = np.array(Eppow_data_list[0])
            EXm = np.array(Empow_data_list[0])
            EXp_2 = np.array(Eppow_data_list[1])
            EXm_2 = np.array(Empow_data_list[1])
            w2 = w**2
            y2 = np.square(y_eval)
            yw = w*y_eval
            est = w2 * (2 * (w2 + y2 - 2*yw * EX) - 2*w2*(1-EX_2)*EXp*EXm - (y2 + w2*EX_2 - 2*yw*EX)*(EXp_2 + EXm_2))
            mean_condVar = np.mean(est)
#             logger.debug(f'w2: {w2} \n y2: {y2} \n yw: {yw}')
#             logger.debug(f'EX: {EX}\n EX2: {EX_2} \n EXp: {EXp} \n EXm: {EXm} \n EXp2: {EXp_2} \n EXm2: {EXm_2}')
#             logger.debug(f'est circ para: {est}')
            if mean_condVar < 0:
                est = w**2 * ((2 - EXp**2 - EXm**2) * np.square(y_eval-w*EX) + 2 * w**2 * (1-EX**2)*(1-EXp*EXm))
                mean_condVar = np.mean(est)
        return mean_condVar
    
    def _estimate_MSEgrad_mean_condVar(self, y_eval, E0pow_data_list, w=1., second_order=False, n0=None,
                                       Scale_para=True, Eppow_data_list=None, Empow_data_list=None,
                                      var_0=None, var_p=None, var_m=None, WoR_correct=False, mini_batch_size=1):
        '''
        解析式をもとに、個別の推定量から E_data[ V[grad[i, data]|data] ]の不偏推定を計算して返す。
        （1/n_shots の係数として使うと、実際の勾配推定量の分散のupper boundとなる）
        Scale_para=True なら scale parameterの勾配の分散を計算
        Epow_data_list[i][j] には、期待値i+1乗の推定量の j-th dataのものを入れる。y_evalのj-th indexと連動させる。
        '''
#         logger = init_logging('EV_estimations', raw_id_for_filename=True)
#         logger.debug(f'E0pow: {E0pow_data_list} \n \n')
#         logger.debug(f'Eppow: {Eppow_data_list} \n \n')
#         logger.debug(f'Eppow: {Empow_data_list} \n \n')
#         logger.debug(f'w: {w} \n \n')
        EX = np.array(E0pow_data_list[0]) #それぞれ、data点をインデックスとしたnumpy array
        EX_2 = np.array(E0pow_data_list[1])
        if WoR_correct:
            N = float(self.num_train_data)
            m = float(mini_batch_size)
            correct_fact = (N - m) * (1. + 1/N) / (N - 1.) + m / N
        else:
            correct_fact = 1.
        if Scale_para:
            EX_3 = np.array(E0pow_data_list[2])
            EX_4 = np.array(E0pow_data_list[3])
            a = -2 * y_eval
            b = 2 * w
                      
            est = a**2 + 4.*a*b*EX + (4.*b**2-a**2)*EX_2 - 4*a*b*EX_3 - 4*b**2*EX_4 #first order
#             logger.debug(f'a: {a} \n b: {b}')
#             logger.debug(f'EX: {EX}\n EX2: {EX_2} \n EX3: {EX_3} \n EX4: {EX_4}')  
#             logger.debug(f'est 1st: {est} \n \n')
            if second_order and n0:
                est += 2*b**2*(1 - 2*EX + EX_2)/(n0-1)
#                logger.debug(f'est +2nd: {est} \n \n')
            mean_condVar = np.mean(est)            
#            logger.debug(f'EV: {mean_condVar} \n \n')
#             Var_X = 1 - np.square(EX)
#             est2 = np.square(a + 2*b * EX) * Var_X + 2*b**2 * np.square(Var_X)
#             logger.debug(f'Var_X: {Var_X}')
#             logger.debug(f'est2: {est2}')
#             logger.debug(f'mean est2: {np.mean(est2)}')
#             mean_condVar = np.mean(est) #これが分散の 1/mn に比例する項の分子の不偏推定量
            if mean_condVar < 0:
                Var_X = var_0 #1 - np.square(EX) ではなく、不偏分散として、n/(n-1)をかけるべき
                est = np.square(a + 2*b * EX) * Var_X 
                if second_order and n0:
                    est += 2*b**2 * np.square(Var_X)/(n0 - 1)
                mean_condVar = np.mean(est)
                #logger.debug(f'biased EV: {mean_condVar} \n \n')
            return mean_condVar
        else:            
            EXp = np.array(Eppow_data_list[0])
            EXm = np.array(Empow_data_list[0])
            EXp_2 = np.array(Eppow_data_list[1])
            EXm_2 = np.array(Empow_data_list[1])
            w2 = w**2
            y2 = np.square(y_eval)
            yw = w*y_eval
            err2 = y2 - 2.*yw*EX + w2 * EX_2
            dpdt2 = w2 * (EXp_2 - 2.*EXp*EXm + EXm_2)
            var_dpdt = w2 * (var_p + var_m)
            #var_dpdt = (var_p + var_m) #上が正しく、これは間違いだが、前を再現
            var_err = w2 * var_0
            est_i = err2 * var_dpdt
            est_0 = var_err * dpdt2            
#           logger.debug(f'w2: {w2} \n y2: {y2} \n yw: {yw} \n \n')
#           logger.debug(f'EX: {EX}\n \n EX2: {EX_2} \n \n EXp: {EXp} \n \n EXm: {EXm} \n \n EXp2: {EXp_2} \n \n EXm2: {EXm_2} \n \n')
#           logger.debug(f'est 1st: {est} \n \n')
            if second_order and n0:
                second_term = var_err*var_dpdt / n0
                est_i += second_term                
            EV_i = np.mean(est_i)
            EV_0 = np.mean(est_0)
            mean_condVar = EV_i + EV_0
            if EV_0 < 0:
                dpdt2 = w2 * np.square(EXp - EXm)
                est_0 = var_err * dpdt2
                EV_0 = np.mean(est_0)
            if EV_i < 0:
                err2 = np.square(y_eval - w * EX)
                est_i = err2 * var_dpdt
                if second_order and n0:
                    est_i += var_err*var_dpdt / n0
                EV_i = np.mean(est_i)
            if mean_condVar < 0:                
                mean_condVar = EV_i + EV_0            
            return mean_condVar, EV_i, EV_0
    
    def _estimate_MSEgrad_Var_condMean(self, y_eval, grad_data_list, size, E0pow_data_list, w=1., Scale_para=True, Eppow_data_list=None, Empow_data_list=None):
        '''
        V_data[ E[grad[i,data]|data] ] のunbiased estimator
        E[grad[i,data]|data]を、data pointをひとつとると確定値が決まる確率変数として分散を推定する。
        不偏分散にはその２乗の推定量がでてくるが、E[grad[i,data]|data]の２乗の不偏推定量を使わないと不偏推定量にならない。
        （サンプル平均の２乗はバイアスがある）
        negativeになることがあるから、その場合は、biasedだが gradの各データ点のサンプル平均の不偏分散をそのまま計算。
        ただし、サンプル平均の不偏分散をそのまま使ったほうが、バイアスはあるが分散は小さくなるようなので、１data点での測定数が多い場合は、
        biasedを使ったほうがよさそう
        '''
        EX_2 = np.array(E0pow_data_list[1]) #それぞれ、data点をインデックスとしたnumpy array
        if Scale_para:
            EX_3 = np.array(E0pow_data_list[2])
            EX_4 = np.array(E0pow_data_list[3])
            a = -2 * y_eval
            b = 2 * w
            grad_data2_list =  a**2 * EX_2 + 2.*a*b * EX_3 + b**2 * EX_4
        else:
            EX = np.array(E0pow_data_list[0])
            EXp = np.array(Eppow_data_list[0])
            EXm = np.array(Empow_data_list[0])
            EXp_2 = np.array(Eppow_data_list[1])
            EXm_2 = np.array(Empow_data_list[1])
            grad_data2_list = (y_eval**2 - 2.*y_eval*w*EX + w**2 * EX_2)* w**2 *(EXp_2 + EXm_2 - 2*EXp*EXm)
        est = np.sum(grad_data2_list)/size - (np.sum(grad_data_list)**2 - np.sum(np.square(grad_data_list)))/(size*(size-1.))
        if est < 0:
            est = np.var(grad_data_list, ddof=1)
        return est
        
    def reset_remain_ind(self):
        self.remain_ind = (np.random.permutation(len(self.x_train))).astype(int)
        #print(self.remain_ind)
        #print('reset_remain_ind')

    def _pop_mini_batch_ind(self, mini_batch_size):
        '''
        self.remain_indに保存してある、ランダムな順に並んだdata点のインデックスから、mini_batch_sizeの分だけpopして返す。
        remain_indを使い尽くしたら、再び並び替える（1 epoch）
        '''
        if len(self.remain_ind) < mini_batch_size:
            eval_ind = np.array(self.remain_ind).copy()
            mini_batch_size -= len(eval_ind)
            self.reset_remain_ind()
            #print(mini_batch_size)
            #print(len(self.remain_ind))
            self.epoch_counter[0] += 1
            eval_ind = np.concatenate((eval_ind, self.remain_ind[:mini_batch_size])).astype(int)
            self.remain_ind = self.remain_ind[mini_batch_size:]
            #print(eval_ind)
            #print('epoch')
        else:
            eval_ind = self.remain_ind[:mini_batch_size]
            #print(eval_ind)
            self.remain_ind = self.remain_ind[mini_batch_size:]
            #print('minibatch')
        return eval_ind
        
    def MSE_grad_eval(self, params, n_shots_list, mini_batch_size=None, option_min_approx=None, option_weight=None, L2_reg_lmd=0.0, Online_reg_w_center=False, reg_w_coef=1, Take_abs=False):        
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
        if Online_reg_w_center:
            if not hasattr(self, '_max_abs_y_online'):
                self._max_abs_y_online = np.max(y_eval)
            else:
                self._max_abs_y_online = max(np.max(y_eval), self._max_abs_y_online)
            w_center = self._max_abs_y_online
        else:
            w_center = self._max_abs_y_train
        L2_reg_cent = np.zeros_like(params)
        L2_reg_cent[-1] = w_center * reg_w_coef

        ############まず、現在のパラメータ（scale parameter の勾配）
        n_shots0 = n_shots_list[-1] #scale parameter の勾配の測定は、現在のパラメータでの期待値だけでいい。そういうパラメータがある場合はこれでいい
        n_shots_data_list = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        w = params[-1]
        #Ep_list_list, obs_var_lists0 = self._sample_obs(params, x_eval, n_shots_data_list, min_approx, order=4, Return_var=True)
        EX0pow_ub_data_list = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, min_approx, order=4, Return_var=False, unbiased=True)
        #obs_var_lists0は、データ点ごとの不偏分散のリスト
        EX0_list = EX0pow_ub_data_list[0]
        #Ep_list_list = [Ob_0_list, Ob_0_list**2, Ep_unbiased_ls_ls[2], Ob_0_list**4] #符号が一致するために、biased estimatorを使う。3乗は符号関係ないのでunbiased
        exp_pred0 = w * EX0_list #weightをかけることに注意 predictionの値
        self.cache_train_pred = exp_pred0
        self.cache_y_train_eval = y_eval

        grad_data_list = 2.*(-y_eval*EX0_list + w * EX0pow_ub_data_list[1]) #各データでのgrad 推定量のリスト。符号は反転してもOKなので、2乗の項は２乗のunbiased estimatorを使う
        grad[-1] = np.mean(grad_data_list)
        #回路のパラメータの勾配
        for i in range(len(params) - 1):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            EXppow_ub_data_list = self._sample_obs_data_by_data(params + ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=False, unbiased=True)
            EXmpow_ub_data_list = self._sample_obs_data_by_data(params - ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=False, unbiased=True)
            EXp_list = EXppow_ub_data_list[0]
            EXm_list = EXmpow_ub_data_list[0]

            grad_data_list = (w * EX0_list - y_eval) * w * (EXp_list - EXm_list)
            grad[i] = np.mean(grad_data_list)        
        sign_for_w = np.full_like(params, 1.)
        if Take_abs:      
            sign_for_w[-1] = np.sign(params[-1])
        return grad + L2_reg_lmd * (params - sign_for_w * L2_reg_cent) #Take_absのとき、weightについては(|w| - a)^2とするため符号をつける。
    
    def MSE_loss_eval(self, params, n_shots, mini_batch_size=None, option_min_approx=None, option_weight=None):
        '''
        与えたパラメータでのミニバッチでそれぞれn_shots使ったコスト関数の推定値を返す。
        '''
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
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]
        n_shots_data_list = np.floor(n_shots*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするバッチだけ抜き出す作用がある。
        EX0pow_ub_list = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, min_approx, order=2, Return_var=False, unbiased=True)
        w = params[-1]
        cost_data_list = w**2 * EX0pow_ub_list[1] - 2. * w * y_eval * EX0pow_ub_list[0] + np.square(y_eval) #weightをかけることに注意
        return np.mean(cost_data_list)
    
    def MSE_iEval(self, params, n_shots_list, option_min_approx=None, option_weight=None, mini_batch_size=None, Max_n_shots0=True, second_order=False, L2_reg_lmd=0.0, Online_reg_w_center=False, reg_w_coef=1.5, WoR_correct=False, Take_abs=False):
        '''
        data_consistent_components=True　も入れてもいいが保留（各成分で違うデータをevalしていいかどうか）
        opt_renormは、re-normalization factorを最適化するかどうか
        n_shots_list は、勾配の各成分のestimationで、ミニバッチ内の全データに使うショット数。同じショット数を分配する。
        WoR_correct: 分散の計算で、without replecement による補正をいれるかどうか: やっぱり意味なさそうなので未実装
        L2_reg_lmd can be a numpy array in general.
        '''
#        logger = init_logging('iEval', raw_id_for_filename=True)
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
        eval_ind = self._pop_mini_batch_ind(size)
        #train dataの重みのミニバッチ取り出し
        wt_dist = np.array(wt_dist)
        wt_dist = wt_dist[eval_ind]
        #train dataからミニバッチを取り出す
        x_eval = self.x_train[eval_ind]
        y_eval = self.y_train[eval_ind]
        grad_data_list = np.zeros((len(params), len(x_eval)))
        if Online_reg_w_center:
            if not hasattr(self, '_max_abs_y_online'):
                self._max_abs_y_online = np.max(y_eval)
            else:
                self._max_abs_y_online = max(np.max(y_eval), self._max_abs_y_online)
            w_center = self._max_abs_y_online
        else:
            w_center = self._max_abs_y_train
        L2_reg_cent = np.zeros_like(params)
        L2_reg_cent[-1] = w_center * reg_w_coef

        ############まず、現在のパラメータ（scale parameter の勾配）
        n0 = n_shots_list[-1]
        if Max_n_shots0:
            n_shots0 = np.max(n_shots_list) #実際の分散が、用いた式以下になることを保証するためにこうする。
        else:
            n_shots0 = n_shots_list[-1]
        n_shots_data_list = np.floor(n_shots0*wt_dist).astype(int) #deterministicにする wt_distは、evaluateするデータだけ抜き出す。
        w = params[-1]
        ##########Ep_list_list, obs_var_lists0 = self._sample_obs(params, x_eval, n_shots_data_list, min_approx, order=4, Return_var=True)
        ############
        if min_approx== -10:
            EX0pow_exact, var_0_exact = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, min_approx, order=4, Return_var=True, unbiased=True)
            #grad 自体はexactにしないで、各データ点の分散だけexactにしてみる
            EX0pow_ub_data_list, var_0 = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, -1, order=4, Return_var=True, unbiased=True)
            grad_data_exact = 2.*(-y_eval*EX0pow_exact[0] + w * EX0pow_exact[1]) #各データでのgrad 推定量のリスト。符号は反転してもOKなので、2乗の項は２乗のunbiased estimatorを使う
        else:
            EX0pow_ub_data_list, var_0 = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, min_approx, order=4, Return_var=True, unbiased=True)
        ###########
        ########obs_var_lists0は、データ点ごとの不偏分散のリスト
        ##############
        EX0_list = EX0pow_ub_data_list[0]
        ##############
        #######Ep_list_list = [Ob_0_list, Ob_0_list**2, Ep_unbiased_ls_ls[2], Ob_0_list**4] #符号が一致するために、biased estimatorを使う。3乗は符号関係ないのでunbiased
        ########################
        exp_pred0 = w * EX0_list #weightをかけることに注意 predictionの値
        self.cache_train_pred = exp_pred0
        self.cache_y_train_eval = y_eval
        grad_data_list[-1] = 2.*(-y_eval*EX0_list + w * EX0pow_ub_data_list[1]) #各データでのgrad 推定量のリスト。符号は反転してもOKなので、2乗の項は２乗のunbiased estimatorを使う
        grad[-1] = np.mean(grad_data_list[-1])
        if min_approx== -10: #debug
            gvar_list_EV[-1] = self._estimate_MSEgrad_mean_condVar(y_eval, EX0pow_exact, w=w, Scale_para=True, var_0=var_0_exact, n0=n0, second_order=second_order)
            gvar_list_b[-1] = self._estimate_MSEgrad_Var_condMean(y_eval, grad_data_exact, size, EX0pow_exact, w=w, Scale_para=True)
        else:
            gvar_list_EV[-1] = self._estimate_MSEgrad_mean_condVar(y_eval, EX0pow_ub_data_list, w=w, Scale_para=True, var_0=var_0, n0=n0, second_order=second_order)
            gvar_list_EV0[-1] = gvar_list_EV[-1]
            gvar_list_b[-1] = self._estimate_MSEgrad_Var_condMean(y_eval, grad_data_list, size, EX0pow_ub_data_list, w=w, Scale_para=True)
        
        ###debug####################
        #M_tot = self.num_train_data
#         n = n_shots_list[-1]
#         outcomes0_data_list, outsum0 = self._sample_outcomes_sum_Z0_all_data(params, x_eval, n_shots_data_list)
#         gEV = 0.
#         b = 2*w
#         m = len(outcomes0_data_list)
#         for i in range(m):
#             a = -2*y_eval[i]
#             outcomes_0 = outcomes0_data_list[i]
#             mean_out = np.mean(outcomes_0)
#             out2 = (n/(n-1.))*mean_out**2 - 1/(n-1.)
#             grad_data_list[-1][i] = 2*(w * out2 - y_eval[i]*mean_out)
#             varO = np.var(outcomes_0,ddof=1)
#             gEV += ((a**2 + 2*b*mean_out)**2 * varO)/m # + 2*b**2*varO**2/(n-1))/m
#         grad[-1] = np.mean(grad_data_list[-1])
#         gvar_list_EV[-1] = gEV
#         gvar_list_b[-1] = np.var(grad_data_list[-1], ddof=1)
        #回路のパラメータの勾配
        for i in range(len(params) - 1):
            #それぞれのデータに何ショット使うかのリスト
            n_shots_data_list = np.floor(n_shots_list[i]*wt_dist).astype(int)
            #######parameter shift rule
            ei = np.zeros_like(params)
            ei[i] = 0.5 * np.pi
            #どっちのシフトにも同じショット数を使う
            ########print(n_shots_data_list)
            if min_approx == -10:
                EXppow_exact, var_p_exact = self._sample_obs_data_by_data(params + ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=True, unbiased=True)
                EXmpow_exact, var_m_exact = self._sample_obs_data_by_data(params - ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=True, unbiased=True)
                EXppow_ub_data_list, var_p = self._sample_obs_data_by_data(params + ei, x_eval, n_shots_data_list, -1, order=2, Return_var=True, unbiased=True)
                EXmpow_ub_data_list, var_m = self._sample_obs_data_by_data(params - ei, x_eval, n_shots_data_list, -1, order=2, Return_var=True, unbiased=True)
                grad_data_exact = (w * EX0pow_exact[0] - y_eval) * w * (EXppow_exact[0] - EXmpow_exact[0])
            else:
                EXppow_ub_data_list, var_p = self._sample_obs_data_by_data(params + ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=True, unbiased=True)
                EXmpow_ub_data_list, var_m = self._sample_obs_data_by_data(params - ei, x_eval, n_shots_data_list, min_approx, order=2, Return_var=True, unbiased=True)
            EXp_list = EXppow_ub_data_list[0]
            EXm_list = EXmpow_ub_data_list[0]

            grad_data_list[i] = (w * EX0_list - y_eval) * w * (EXp_list - EXm_list)
            grad[i] = np.mean(grad_data_list[i])
            if min_approx== -10:
                gvar_list_EV[i] = self._estimate_MSEgrad_mean_condVar(y_eval, EX0pow_exact, w=w, second_order=second_order, n0=n0, var_0=var_0_exact,
                                                                       Scale_para=False, Eppow_data_list=EXppow_exact,
                                                                      Empow_data_list=EXmpow_exact, var_p=var_p_exact, var_m=var_m_exact)
                gvar_list_b[i] = self._estimate_MSEgrad_Var_condMean(y_eval, grad_data_exact, size, EX0pow_exact, w=w,
                                                                      Scale_para=False, Eppow_data_list=EXppow_exact, Empow_data_list=EXmpow_exact)
            else:
                EV, EVi, EV0 = self._estimate_MSEgrad_mean_condVar(y_eval, EX0pow_ub_data_list, w=w, second_order=second_order, n0=n0, var_0=var_0,
                                                                       Scale_para=False, Eppow_data_list=EXppow_ub_data_list,
                                                                      Empow_data_list=EXmpow_ub_data_list, var_p=var_p, var_m=var_m)
                gvar_list_EV[i] = EV
                gvar_list_EVi[i] = EVi
                gvar_list_EV0[i] = EV0
                gvar_list_b[i] = self._estimate_MSEgrad_Var_condMean(y_eval, grad_data_list, size, EX0pow_ub_data_list, w=w,
                                                                      Scale_para=False, Eppow_data_list=EXppow_ub_data_list, Empow_data_list=EXmpow_ub_data_list)
            
            ###debug###############
#             ei = np.zeros_like(params)
#             ei[i] = 0.5 * np.pi
            
#             outcomes_p_data_list, outsum_p = self._sample_outcomes_sum_Z0_all_data(params + ei, x_eval, n_shots_data_list)
#             outcomes_m_data_list, outsum_m = self._sample_outcomes_sum_Z0_all_data(params - ei, x_eval, n_shots_data_list)
#             gEV = 0.
#             for j in range(len(outcomes_p_data_list)):
#                 outcomes_0 = outcomes0_data_list[j]
#                 outcomes_p = outcomes_p_data_list[j]
#                 outcomes_m = outcomes_m_data_list[j]
#                 M0 = np.mean(outcomes_0)
#                 Mp = np.mean(outcomes_p)
#                 Mm = np.mean(outcomes_m)
#                 grad_data_list[i][j] = (w * M0 - y_eval[j]) * w * (Mp - Mm)
#                 M02 = (n/(n-1.))*M0**2 - 1/(n-1.)
#                 var_err_mean2 = (w**2*np.var(outcomes_0, ddof=1) + y_eval[j]**2)* w**2 * (Mp - Mm)**2
#                 var_df_mean2 = w**2 * (np.var(outcomes_p,ddof=1) + np.var(outcomes_m,ddof=1))*(w * M0 - y_eval[j])**2
#                 varvar_ov_n0 = w**2*(np.var(outcomes_p,ddof=1) + np.var(outcomes_m,ddof=1))*(w**2*np.var(outcomes_0,ddof=1) + y_eval[j]**2)/n_shots_list[-1]
#                 #total_var = np.var((w * outcomes_0 - y_eval[i]) * w * (outcomes_p - outcomes_m), ddof=1)
#                 total_var_mod = var_err_mean2 + var_df_mean2 # + varvar_ov_n0
#                 gEV += total_var_mod / m
#                 #logger.debug(f'{i}-th data grad outcomes {(w * outcomes_0 - y_eval[i]) * w * (outcomes_p - outcomes_m)}')
#                 #logger.debug(f'{i}-th data (w * outcomes_0 - y_eval[i]) outcomes {(w * outcomes_0 - y_eval[i])}')
#                 if total_var_mod > 100:
#                     logger.debug(f'{j}-th data var(w * outcomes_0 - y_eval[i]) * mean**2 {var_err_mean2}')
#                     if var_err_mean2 > 100:
#                         logger.debug('var_err_mean2 over 100')
#                     #logger.debug(f'{i}-th data w * (outcomes_p - outcomes_m) outcomes {w * (outcomes_p - outcomes_m)}')
#                     logger.debug(f'{j}-th data var(w * (outcomes_p - outcomes_m))* mean**2 {var_df_mean2}')
#                     if var_df_mean2 > 100:
#                         logger.debug('var_df_mean2 over 100')
#                     logger.debug(f'{j}-th data var*var {varvar_ov_n0}')
#                     #logger.debug(f'{i}-th data grad variance {total_var}')
#                     logger.debug(f'{j}-th data total grad variance modified {total_var_mod}')
#                 logger.debug('------------------------------------------------------------------------------------------------------')
#             grad[i] = np.mean(grad_data_list[i])
#             gvar_list_EV[i] = gEV
#             gvar_list_b[i] = np.var(grad_data_list[i], ddof=1)
            ##############
        #trackする変数名でiEval_returnをtrackして、grad_data_listも記録できるようにした。データごとの勾配ベクトルたちが成分間に相関をもっているかどうか見るため。
        sign_for_w = np.full_like(params, 1.)
        if Take_abs:
            sign_for_w[-1] = np.sign(params[-1])
        output = {
            'grad': grad + L2_reg_lmd * (params - L2_reg_cent * sign_for_w), #regularizationを加える
            'gvar_list_EV':gvar_list_EV,
            'gvar_list_EVi':gvar_list_EVi,
            'gvar_list_EV0':gvar_list_EV0,
            'gvar_list_b':gvar_list_b, 
            'grad_data_list':grad_data_list
        }
        return output
    
    # refoqusで使うために、全体の分散をサンプルから求めてしまうやつ。data点についてもランダムにサンプルを取っているときに使える    
    def _direct_sample_MSEgrad_varUB_whole(self, y_eval, outcomes_whole_no_shift, w=1, Scale_para=False, outcomes_whole_p=None, outcomes_whole_m=None):
        '''
        outcomes_whole はそれぞれ、[i][j]要素に、i-th dataでの、j番目の測定のoutcomeが入っている。(各iにoutcomesのarrayが入っているとする。)
        '''
        M_tot = self.num_train_data
        if Scale_para:
            out1_list, out2_list = zip(*[split_array_equally(outcomes) for outcomes in outcomes_whole_no_shift])
            out1_array = np.concatenate(out1_list)
            out2_array = np.concatenate(out2_list)
            repeat_counts = [len(outcomes) for outcomes in out1_list]
            grad_outcomes_data = 2 * (w * out1_array - np.repeat(y_eval, repeat_counts)) * out2_array
        else:
            outcomes_0 = np.concatenate(outcomes_whole_no_shift)
            outcomes_p = np.concatenate(outcomes_whole_p)
            outcomes_m = np.concatenate(outcomes_whole_m)
            repeat_counts = [len(outcomes) for outcomes in outcomes_whole_no_shift]
            grad_outcomes_data = (w * outcomes_0 - np.repeat(y_eval, repeat_counts)) * w * (outcomes_p - outcomes_m)

        return np.var(grad_outcomes_data, ddof=1)

    def MSE_iEval_iRandom(self, params, n_shots_list, option_weight=None, L2_reg_lmd=0.0, L2_reg_cent=None, reg_w_coef=1.5, Take_abs=False):
        '''
        gradの各成分独立にdata点にショット数振り分けてサンプリングする。independent Random (iRandom) data sampling 
        (mini-batch ではないdata点の取り方として)
        先行研究の refoqus (gCANS) に使うため
        '''
        grad = np.zeros_like(params)
        gvar_list = np.zeros_like(params)

        if option_weight is None:
            wt_dist = self._uniform_wt_dist.copy()
        else:
            wt_dist = option_weight.copy()
            wt_dist = np.array(wt_dist) / np.sum(wt_dist) #確率分布にする。
        
        w_center = self._max_abs_y_train
        L2_reg_cent = np.zeros_like(params)
        L2_reg_cent[-1] = w_center * reg_w_coef

        w = params[-1]
        M_tot = self.num_train_data
        for i in range(len(params)):
            n_shots_data_list = np.random.multinomial(n_shots_list[i], wt_dist)
            x_eval = self.x_train[n_shots_data_list != 0]
            y_eval = self.y_train[n_shots_data_list != 0]
            n_shots_eval = n_shots_data_list[n_shots_data_list != 0]
            n = n_shots_list[i]
            if i == (len(params) - 1): #scale parameter w の勾配
                n_shots_eval *= 2 #2乗があるので、１つの推定に２ショット使うとして２倍する。
                outcomes0_data_list, outsum = self._sample_outcomes_sum_Z0_all_data(params, x_eval, n_shots_eval)
                #n_shots_eval は 2s_j になっていて、s_j が multinomial of nで決まるから、E[2s_j(2s_j-1)]=4p_j^2*n(n-1)+2p_j*n
#                 if self.min_approx_shots == -10:
#                     EX0pow_exact, var_0_exact = self._sample_obs_data_by_data(params, x_eval, n_shots_data_list, -10, order=1, Return_var=True, unbiased=True)
#                     EX0 = EX0pow_exact[0]
#                     grad[i] =
#                else:
                #2s_i を使っているので、分母は、E[2s(2s-1)]=4p^2*n(n-1) + 2np
                grad[i] = np.sum(w * (outsum**2 - n_shots_eval)/(2*n*(n-1)/M_tot + n) - y_eval * outsum / n) #2倍は分母と約してある
                gvar_list[i] = self._direct_sample_MSEgrad_varUB_whole(y_eval, outcomes0_data_list, w=w, Scale_para=True)
                #exp_pred0 = w * EX0pow_list[0] #weightをかけることに注意 predictionの値
                #self.cache_train_pred = exp_pred0
            else: #回路パラメータ
                outcomes0_data_list, outsum0 = self._sample_outcomes_sum_Z0_all_data(params, x_eval, n_shots_eval)
                ei = np.zeros_like(params)
                ei[i] = 0.5 * np.pi
                outcomes_p_data_list, outsum_p = self._sample_outcomes_sum_Z0_all_data(params + ei, x_eval, n_shots_eval)
                outcomes_m_data_list, outsum_m = self._sample_outcomes_sum_Z0_all_data(params - ei, x_eval, n_shots_eval)
                grad[i] = np.sum((w * outsum0/(n + n*(n-1)/M_tot) - y_eval/n) * (outsum_p - outsum_m) * w) #積の取り扱い。各outcomesを、1/pして、和を取るときには重み倍だから、M_totがでる。
                gvar_list[i] = self._direct_sample_MSEgrad_varUB_whole(y_eval, outcomes0_data_list,
                                                                       w=w, Scale_para=False,outcomes_whole_p=outcomes_p_data_list, outcomes_whole_m=outcomes_m_data_list)
        sign_for_w = np.full_like(params, 1.)
        if Take_abs:
            sign_for_w[-1] = np.sign(params[-1])
        iEval_return = {'grad':grad + L2_reg_lmd * (params - L2_reg_cent * sign_for_w), 'gvar_list':gvar_list}
        return iEval_return
    
    def generate_labels(self, x_data, target_params):
        '''
        target_paramsが真のパラメータとなるように
        x_dataたちをQNNに与えたときのoutputたちをデータとして生成する。
        '''
        global generate_labels_flag
        generate_labels_flag = True
        return self.exact_predict(x_data, target_params) 

    @classmethod
    def grid_search_kCV(cls, optimizer_name, circuit, n_qubits, x_data, y_data, grid_para, fixed_para=None, loss='mean_squared_error', options=None,
                        n_splits=5, parallel=False, processes=-1, select_min_score=None,
                        min_approx_shots=-1, n_params=None, init_params=None, Random_search=False, random_num=10, 
                        init_spec_dict=None, manual_methods_setting_dict=None, score_evaluator=None, random_seed_data = 42, use_hpm=False,
                        shot_time=1.0, per_circuit_overhead=0.0, communication_overhead=0.0, return_all_results=True, Max_init_w_center=False, init_w_sgm=0., **kwargs):
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
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        if select_min_score is None:
            if loss == 'mean_squared_error':
                select_min_score = True
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
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
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
        rng = np.random.default_rng()
        def single_proc(args, idx):
            ##logging.basicConfig(level=#logging.DEBUG, filename='apps.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')            
            x_train = x_data[train_inds] #train data として使うデータたち（複数）のインデックスたちのリストが指定される
            y_train = y_data[train_inds]
            x_test = x_data[test_inds]
            y_test = y_data[test_inds]
            if init_params is None:
                if Max_init_w_center:
                    init_w_center = np.max(np.abs(y_train))
                else:
                    init_w_center = 1
                init_para = np.random.uniform(0.0, 2*np.pi, n_params)
                init_para[-1] = rng.normal(loc=init_w_center, scale=init_w_sgm)
            else:
                init_para = init_params
            file_id = str(args['para_set_id']) + '_' + str(args['dataset_id']) + '_' + idstr
            ##logging.debug('just before construct qnn')
            qnn = cls(n_qubits, circuit, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, file_id=file_id,
                            min_approx_shots=min_approx_shots, shot_time=shot_time, per_circuit_overhead=per_circuit_overhead, communication_overhead=communication_overhead)
            hpara = fixed_para.copy()
            hpara.update(args['search_para'])
            #logging.debug('just before compile')
            qnn.compile(optimizer_name, loss, hpara=hpara, init_params=init_para, 
                        init_spec_dict=init_spec_dict, manual_methods_setting_dict=manual_methods_setting_dict)
            results = qnn.fit(options=options, auto_save=False, process_id=args, **kwargs)
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
                        parallel=False, processes=-1, min_approx_shots=-1, n_params=None, random_seed_data=42,                      
                        manual_methods_setting_dict=None, save_file_name=None, save_directory=None, optimizer_str_map=None, Max_init_w_center=False, init_w_sgm=0.,
                        shot_time=1.0, per_circuit_overhead=0.0, communication_overhead=0.0):
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
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                for i in range(num_trials_per_combination):                    
                    train_test_inds.extend(kf.split(x_data))
                num_trials_per_combination = 1
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                train_test_inds = kf.split(x_data)
        if n_params is None:
            n_params = len(circuit.get_parameters()) + 1
        #multiprocessingのときは必要
        if processes < 0:
            processes = cpu_count() - 1
        base_qnn = myQNN_regression(n_qubits, circuit, min_approx_shots=min_approx_shots, 
                                    setting_dict=manual_methods_setting_dict, shot_time=shot_time, per_circuit_overhead=per_circuit_overhead, communication_overhead=communication_overhead)
        opt_trials = OptimizationTrialsManager(base_qnn, x_data, y_data)
        #各試行のデータを準備する。
        opt_trials.bulk_add_trials(optimizers, train_test_inds, num_trials_per_combination, init_params, hpara_dict, optimization_options, compile_options, fit_kwargs_dict, optimizer_str_map=optimizer_str_map, Max_init_w_center=Max_init_w_center, init_w_sgm=init_w_sgm)
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
            dill.dump((self._base_qnn, self._base_x_data, self._base_y_data, self.trials, self.metadata), file)

    def load(self, filename=None, directory=None):
        load_dir = directory if directory else ''
        load_filename = filename if filename else self.default_filename
        load_path = os.path.join(load_dir, load_filename)

        with open(load_path, 'rb') as file:
            self._base_qnn, self._base_x_data, self._base_y_data, loaded_trials, self.metadata = dill.load(file)
            
            for trial_name, trial_data in loaded_trials.items():
                self.trials[trial_name] = trial_data

    def bulk_add_trials(self, optimizers, data_splits=None, num_trials_per_combination=1, init_params=None, hpara_dict=None, optimization_options=None, compile_options=None, kwargs_dict=None, 
                        optimizer_str_map=None, Max_init_w_center=False, init_w_sgm=0.):
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
        optimizerクラス名以外の名前で optimizers を指定して、同じoptimizerの別の設定を比較したり、都合のいい名前にしたりできる。
        例えば、optimizers = ['Adam100', 'Adam1000'], optimizer_str_map = {'Adam100': 'Adam', 'Adam1000': 'Adam'}
        のようにして、optionsやhparaを、'Adam100'の項目に指定すれば良い。optimizer_str_mapで、対応する実行するoptimizerのクラス名を指定。
        こうすれば、任意の名前'Adam100'などをoptimizer名として扱うことができる。
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
        rng = np.random.default_rng()

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
                        if Max_init_w_center:
                            init_w_center = np.max(np.abs(self._base_y_data[train_inds]))
                        else:
                            init_w_center = 1
                        if (trial_num, split_id) not in init_params_cache:
                            init_para = np.random.uniform(-np.pi, np.pi, n_params)
                            init_para[-1] = rng.normal(loc=init_w_center, scale=init_w_sgm)
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
                                    init_params_cache[(trial_num, split_id)] = np.random.uniform(0, 2*np.pi, n_params)
                                params = init_params_cache[(trial_num, split_id)]
                        elif depth == 3:
                            if isinstance(init_params[trial_num], (list, np.ndarray)):
                                params = init_params[trial_num][split_id]
                            elif optimizer in init_params[trial_num]:
                                params = init_params[trial_num][optimizer]
                            else:
                                if (trial_num, split_id) not in init_params_cache:
                                    init_params_cache[(trial_num, split_id)] = np.random.uniform(0, 2*np.pi, n_params)
                                params = init_params_cache[(trial_num, split_id)]
                        elif depth == 4 and optimizer in init_params[trial_num][split_id]:
                            params = init_params[trial_num][split_id][optimizer]
                        else:
                            if (trial_num, split_id) not in init_params_cache:
                                init_params_cache[(trial_num, split_id)] = np.random.uniform(0, 2*np.pi, n_params)
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
import matplotlib.cm as cm

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


    def cv_all_trials(self, x_label='total_shots', y_labels=None, stat_proc="mean_and_median", quantile_mergin=0.25, end=None):
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
                    mean_df, spread_df_mean = self.stat_proc_data(results_list, x_label, y_label, "mean", quantile_mergin=quantile_mergin, end=end)
                    if optimizer + key_suffix not in self.cv_mean:
                        self.cv_mean[optimizer + key_suffix] = {}
                    self.cv_mean[optimizer + key_suffix] = {"stat": mean_df, "spread": spread_df_mean}

                if stat_proc in ["median", "mean_and_median"]:
                    median_df, spread_df_median = self.stat_proc_data(results_list, x_label, y_label, "median", quantile_mergin=quantile_mergin, end=end)
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

    def stat_proc_data(self, results_list, x_label='total_shots', y_label='func_evol', stat_proc="median", unit=1, end=None, quantile_mergin=0.25):
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
                upper = grouped_df.apply(lambda group: group.quantile(1-quantile_mergin, axis=1))
                lower = grouped_df.apply(lambda group: group.quantile(quantile_mergin, axis=1))
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
                upper = df.quantile(1-quantile_mergin, axis=1)
                lower = df.quantile(quantile_mergin, axis=1)
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
                        single_split_data=False, split_id=0, save_figure=False, directory=None, file_extension='svg', xlim=None, ylim=None, xlog=False, ylog=False, label_mapping=None, yticks=None):
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
                    if y_label not in results and y_label != 'scale_para':
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
                    
                    if y_label == 'scale_para':
                        data = [para[-1] for para in results['para_evol']]
                    else:
                        data = [entry[data_type] if isinstance(entry, dict) else entry for entry in results[y_label]]
                    optimizer_w_suff = f'{optimizer} {y_legend_suffix}'
                    if label_mapping and optimizer in label_mapping:
                        optimizer_w_suff = label_mapping[optimizer]
                    label = f"{optimizer_w_suff} ({data_type})"

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
            if yticks is not None:
                ax.set_yticks(yticks)
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
                        save_figure=False, directory=None, file_extension='svg', xlim=None, ylim=None, xlog=False, ylog=False, label_mapping=None, yticks=None):
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
                    if "_" in optimizer_w_suff:
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
                                ax.fill_between(mean_data.index, lower_bound, upper_bound, color=color, alpha=0.2)
                                created_labels.add(label)
                            else:
                                ax.plot(mean_data, color=color, linestyle=line_style)
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
