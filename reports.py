from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess


def create_models():
    models = {
        'BayesianRidge': BayesianRidge(),
        'LinearRegression': LinearRegression(),
        'ElasticNet': ElasticNet(),
        'SVR(rbf)': SVR(kernel='rbf'),
        'SVR(linear)': SVR(kernel='linear'),
        'Lasso': Lasso(),
        'GBR': GradientBoostingRegressor(n_estimators=300, max_depth=3),
    }
    return models


def compare_models(models, inputs, report_dir, n_folds=6):
    """

    :param dict models: 模型
    :param dict inputs: 输入文件
    :param report_dir: 输出报告文件夹
    :param int n_folds: 设置交叉检验的次数
    :return:
    """
    model_names = list(models.keys())
    model_lists = list(models.values())
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score]
    with open(report_dir + '/report.md', 'w') as fh:
        fh.write('# Model Comparison Report\n\n')
        for inp in inputs:
            fp = inputs[inp]
            x, y = preprocess.load_data(fp)
            n_samples, n_features = x.shape
            cv_score_list = []  # k-fold 交叉检验结果列表
            loo_score_list = []  # LOO 交叉检验结果列表
            pre_y_list = []  # 各个回归模型预测的y值列表
            # 交叉验证
            for model in model_lists:
                # k-folds
                scores = -cross_val_score(model, x, y, cv=n_folds, scoring='neg_mean_absolute_error')
                cv_score_list.append(scores)
                # LOO
                scores2 = -cross_val_score(model, x, y, cv=x.shape[0], scoring='neg_mean_absolute_error')
                loo_score_list.append(np.mean(scores2))
                pre_y_list.append(model.fit(x, y).predict(x))
            # 模型效果指标评估
            model_metrics_list = []
            for i in range(len(model_names)):
                tmp_list = []
                for m in model_metrics_name:
                    tmp_score = m(y, pre_y_list[i])
                    tmp_list.append(tmp_score)
                model_metrics_list.append(tmp_list)
            # 建立交叉检验的数据框
            df_cv = pd.DataFrame(cv_score_list, index=model_names)
            df_loo = pd.DataFrame(loo_score_list, index=model_names, columns=['Mean MAE'])
            # 建立回归指标的数据框
            df_me = pd.DataFrame(model_metrics_list, index=model_names, columns=['EV', 'MAE', 'MSE', 'MedAE', 'R2'])
            # 输出图像
            # 预测对比
            plt.figure(figsize=(12, 6))
            plt.plot(np.arange(x.shape[0]), y, color='k', label='Real Height')
            for i, pre_y in enumerate(pre_y_list):
                plt.plot(np.arange(x.shape[0]), pre_y, label=model_names[i])
            plt.title('Regression result comparison for %s' % inp)
            plt.legend(loc='upper right')
            plt.ylabel('real and predicted value')
            plt.savefig(report_dir + '/fig_predict_%s.jpg' % inp)
            # 交叉检验
            plt.figure(figsize=(12, 6))
            for i in range(len(model_names)):
                plt.plot(np.arange(n_folds), df_cv.loc[model_names[i]], label=model_names[i])
            plt.title('%d folds cross validation scores' % n_folds)
            plt.legend(loc='lower right')
            plt.ylim(0, 10)
            plt.ylabel('MAE')
            plt.savefig(report_dir + '/fig_k_folds_cv_%s.jpg' % inp)
            # 回归指标图
            plt.figure(figsize=(18, 6))
            plt.subplot(131)
            plt.bar(np.arange(len(model_names)), df_loo['Mean MAE'], label='Mean MAE for LOO')
            for i in range(len(model_names)):
                v = df_loo['Mean MAE'][i]
                plt.text(i, v + 0.05, '%.4f' % v, ha='center', va='bottom', fontsize=7)
            plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
            plt.title('Mean absolute error for LOO %s' % inp)
            plt.ylim(0, 8)
            plt.ylabel('mean absolute error')
            plt.subplot(132)
            plt.bar(np.arange(len(model_names)), df_me['MAE'].values, label='MAE')
            for i in range(len(model_names)):
                v = df_me['MAE'][i]
                plt.text(i, v + 0.05, '%.4f' % v, ha='center', va='bottom', fontsize=7)
            plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
            plt.title('Mean absolute error for %s' % inp)
            plt.ylabel('mean absolute error')
            plt.subplot(133)
            plt.bar(np.arange(len(model_names)), df_me['R2'].values, label='R2')
            for i in range(len(model_names)):
                v = df_me['R2'][i]
                plt.text(i, v + 0.01, '%.4f' % v, ha='center', va='bottom', fontsize=7)
            plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
            plt.title('R2 for %s' % inp)
            plt.ylabel('R2')
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(report_dir + '/fig_metrics_%s.jpg' % inp)
            # 输出报告
            fh.write('## Input %s\n\n' % inp)
            fh.write('samples: %d \t features: %d\n\n' % (n_samples, n_features / 3))
            fh.write('### Cross validation result\n\n')
            fh.write('#### %d-folds CV\n\n' % n_folds)
            fh.write(output_dataframe(df_cv, 'Model'))
            fh.write('\n\n')
            fh.write('#### LOO\n\n')
            fh.write(output_dataframe(df_loo, 'Model'))
            fh.write('\n\n')
            fh.write('### Regression metrics\n\n')
            fh.write(output_dataframe(df_me, 'Model'))
            fh.write('\n\n')
        # 报告通用信息
        fh.write('## Remark\n\n')
        fh.write('short name | full name\n')
        fh.write('-----------|----------\n')
        fh.write('EV | explained variance\n')
        fh.write('MAE | mean absolute error\n')
        fh.write('MSE | mean squared error\n')
        fh.write('MedAE | median absolute error\n')
        fh.write('R2 | coefficient of determination\n')


def output_dataframe(dataframe, name=''):
    content = []
    cols = name + ' | ' + ' | '.join([str(i) for i in dataframe.columns])
    content.append(cols)
    n_cols = len(dataframe.columns)
    h = '|'.join([5 * '-' for i in range(n_cols + 1)])
    content.append(h)
    for index, row in dataframe.iterrows():
        r = index + ' | ' + ' | '.join([str(i) for i in row])
        content.append(r)
    return '\n'.join(content)


def main():
    models = create_models()
    inputs = {
        # 'snp_1996': 'tmp/train_1996.csv',
        'snp_119': 'tmp/train_119.csv',
        'snp_100': 'tmp/train_100.csv',
        'snp_43': 'tmp/train_43.csv',
    }
    compare_models(models=models, inputs=inputs, report_dir='report', n_folds=10)


if __name__ == '__main__':
    main()
