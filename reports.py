from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import preprocess


def create_models():
    models = {
        'BayesianRidge': BayesianRidge(),
        # 'LinearRegression': LinearRegression(),
        'ElasticNet': ElasticNet(),
        'SVR(rbf)': SVR(kernel='rbf'),
        'SVR(linear)': SVR(kernel='linear'),
        'Lasso': Lasso(),
        'GBR': GradientBoostingRegressor(n_estimators=300, max_depth=3),
    }
    return models


def compare_models(models, inputs, report_dir, n_folds=6, loo=False):
    """

    :param dict models: 模型
    :param dict inputs: 输入文件
    :param report_dir: 输出报告文件夹
    :param int n_folds: 设置交叉检验的次数
    :param bool loo: 是否使用 LOO 交叉检验
    :return:
    """
    model_names = list(models.keys())
    model_lists = list(models.values())
    with open(report_dir + '/report.md', 'w') as fh:
        fh.write('# Model Comparison Report\n\n')
        for inp in inputs:
            print('Start analyze %s' % inp)
            fp = inputs[inp]
            print('Load data')
            x, y = preprocess.load_data(fp, True)  # 有性别
            xg, yg = preprocess.load_data(fp, False)  # 无性别
            n_samples, n_features = x.shape
            print('Samples %d, Features %d' % (n_samples, n_features))
            print('Start cross validation')
            cv_score_list, loo_score_list, loost_score_list = get_cross_validation_score(model_lists, x, y, n_folds, loo)
            cv_g_score_list, loo_g_score_list, loost_g_score_list = get_cross_validation_score(model_lists, xg, yg, n_folds, loo)
            print('Start predict')
            pre_y_list = get_predicted(model_lists, x, y)
            pre_y_g_list = get_predicted(model_lists, xg, yg)
            print('Calculate metrics')
            model_metrics_list = get_model_metrics(model_names, y, pre_y_list)
            model_metrics_g_list = get_model_metrics(model_names, yg, pre_y_g_list)
            # 建立数据框
            df_cv = pd.DataFrame(cv_score_list, index=model_names)
            df_cv_g = pd.DataFrame(cv_g_score_list, index=model_names)
            loo_columns = ['Mean MAE', 'Var MAE']
            metrics_columns = ['EV', 'MAE', 'MSE', 'MedAE', 'R2']
            if loo:
                df_loo = pd.DataFrame(loost_score_list, index=model_names, columns=loo_columns)
                df_loo_g = pd.DataFrame(loost_g_score_list, index=model_names, columns=loo_columns)
            else:
                df_loo = None
                df_loo_g = None
            df_me = pd.DataFrame(model_metrics_list, index=model_names, columns=metrics_columns)
            df_me_g = pd.DataFrame(model_metrics_g_list, index=model_names, columns=metrics_columns)
            # 输出图像
            print('Plot images %s' % inp)
            img_real_dis = plot_real_y_hist(report_dir, inp, y, pd.read_csv(fp, index_col=0))
            img_comp = plot_predict_comparison(report_dir, inp + '_with_gender', model_names, y, pre_y_list)
            img_comp_g = plot_predict_comparison(report_dir, inp + '_without_gender', model_names, yg, pre_y_g_list)
            img_k_folds, img_loo_dis, img_loo_mae = plot_cross_val(report_dir, inp + '_with_gender', model_names, df_cv, n_folds, loo, loo_score_list, df_loo)
            img_k_folds_g, img_loo_dis_g, img_loo_mae_g = plot_cross_val(report_dir, inp + '_without_gender', model_names, df_cv_g, n_folds, loo, loo_g_score_list, df_loo_g)
            img_me = plot_metrics(report_dir, inp + '_with_gender', model_names, df_me)
            img_me_g = plot_metrics(report_dir, inp + '_without_gender', model_names, df_me_g)
            plt.close('all')
            # 输出报告
            print('Output to report %s' % inp)
            fh.write('## Input %s\n\n' % inp)
            fh.write('samples: %d \t features: %d (%d genotypes and gender)\n\n'
                     % (n_samples, n_features / 3, n_features / 3 - 1))
            fh.write('### Sample distribution\n\n')
            fh.write('![real distribution](%s)\n\n' % img_real_dis)
            fh.write('![real predict comparison with gender](%s)\n\n' % img_comp)
            fh.write('![real predict comparison without gender](%s)\n\n' % img_comp_g)
            fh.write('### Cross validation result\n\n')
            fh.write('#### %d-folds CV MAE with gender\n\n' % n_folds)
            fh.write(output_dataframe(df_cv, 'Model'))
            fh.write('\n\n')
            fh.write('![k-folds mae with gender](%s)\n\n' % img_k_folds)
            fh.write('#### %d-folds CV MAE without gender\n\n' % n_folds)
            fh.write(output_dataframe(df_cv_g, 'Model'))
            fh.write('\n\n')
            fh.write('![k-folds mae without gender](%s)\n\n' % img_k_folds_g)
            if loo:
                fh.write('#### LOO MAE with gender\n\n')
                fh.write(output_dataframe(df_loo, 'Model'))
                fh.write('\n\n')
                fh.write('![loo mae with gender](%s)\n\n' % img_loo_mae)
                fh.write('![loo distribution with gender](%s)\n\n' % img_loo_dis)
                fh.write('#### LOO MAE without gender\n\n')
                fh.write(output_dataframe(df_loo_g, 'Model'))
                fh.write('\n\n')
                fh.write('![loo mae without gender](%s)\n\n' % img_loo_mae_g)
                fh.write('![loo distribution without gender](%s)\n\n' % img_loo_dis_g)
            fh.write('### Regression metrics with gender\n\n')
            fh.write(output_dataframe(df_me, 'Model'))
            fh.write('\n\n')
            fh.write('![metrics with gender](%s)\n\n' % img_me)
            fh.write('### Regression metrics without gender\n\n')
            fh.write(output_dataframe(df_me_g, 'Model'))
            fh.write('\n\n')
            fh.write('![metrics without gender](%s)\n\n' % img_me_g)
        # 报告通用信息
        fh.write('## Remark\n\n')
        fh.write('short name | full name\n')
        fh.write('-----------|----------\n')
        fh.write('LOO | leave one out\n')
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


def get_model_metrics(model_names, y, pre_y_list):
    """
    模型效果指标评估

    :param model_names:
    :param y:
    :param pre_y_list:
    :return:
    """
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error,
                          r2_score]
    model_metrics_list = []
    for i in range(len(model_names)):
        tmp_list = []
        for m in model_metrics_name:
            tmp_score = m(y, pre_y_list[i])
            tmp_list.append(tmp_score)
        model_metrics_list.append(tmp_list)
    return model_metrics_list


def get_cross_validation_score(model_lists, x, y, n_folds, loo=False):
    """
    获取各个模型的交叉检验结果

    :param model_lists:
    :param x:
    :param y:
    :param n_folds:
    :param loo:
    :return:
    """
    cv_score_list = []  # k-fold 交叉检验结果列表
    loo_score_list = []  # LOO 交叉检验结果列表
    loost_score_list = []  # LOO 交叉检验统计结果列表
    for model in model_lists:
        # k-folds
        scores = -cross_val_score(model, x, y, cv=n_folds, scoring='neg_mean_absolute_error')
        cv_score_list.append(scores)
        # LOO
        if loo:
            scores2 = -cross_val_score(model, x, y, cv=x.shape[0], scoring='neg_mean_absolute_error')
            loo_score_list.append(scores2)
            loost_score_list.append([np.mean(scores2), np.var(scores2)])
    return cv_score_list, loo_score_list, loost_score_list


def get_predicted(model_lists, x, y):
    """
    获取各个模型的预测结果

    :param model_lists:
    :param x:
    :param y:
    :return:
    """
    pre_y_list = []  # 各个回归模型预测的y值列表
    for model in model_lists:
        pre_y_list.append(model.fit(x, y).predict(x))
    return pre_y_list


def plot_real_y_hist(img_dir, name, y, dataframe):
    """
    真实身高直方图

    :param img_dir:
    :param name:
    :param y:
    :param dataframe
    :return:
    """
    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.hist(y, bins=40)
    plt.title('Real Height')
    plt.xlabel('Height(cm)')
    plt.ylabel('Amount')
    plt.subplot(212)
    total = dataframe['height']
    male = dataframe[dataframe['gender'] == 1]['height']
    female = dataframe[dataframe['gender'] == 0]['height']
    df = pd.DataFrame({'Male': male, 'Female': female, 'Total': total})
    df.boxplot()
    plt.title('Real height by gender')
    plt.ylabel('Height(cm)')
    plt.subplots_adjust(hspace=0.3)
    imgname = 'fig_real_height_%s.jpg' % name
    plt.savefig(img_dir + '/' + imgname)
    return imgname


def plot_predict_comparison(img_dir, name, model_names, y, pre_y_list):
    """
    预测对比图

    :param img_dir:
    :param name:
    :param model_names:
    :param y:
    :param pre_y_list:
    :return:
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y, y, '-', color='k')
    styles = ['o', '*', '+', ',', 'v', '^', '<', '>']
    for i, pre_y in enumerate(pre_y_list):
        plt.plot(y, pre_y, styles[i], label=model_names[i])
    plt.title('Regression result comparison for %s' % name)
    plt.legend(loc='upper right')
    plt.xlabel('real height')
    plt.ylabel('predicted height')
    imgname = 'fig_comp_%s.jpg' % name
    plt.savefig(img_dir + '/' + imgname)
    return imgname


def plot_cross_val(img_dir, name, model_names, df_cv, n_folds, loo=False, loo_score_list=None, df_loo=None):
    """
    交叉检验图

    :param img_dir:
    :param name:
    :param model_names:
    :param df_cv:
    :param n_folds:
    :param loo:
    :param loo_score_list:
    :param df_loo
    :return:
    """
    # k-folds cv
    plt.figure(figsize=(12, 6))
    for i in range(len(model_names)):
        plt.plot(np.arange(n_folds), df_cv.loc[model_names[i]], label=model_names[i])
    plt.title('%d folds cross validation scores' % n_folds)
    plt.legend(loc='lower right')
    plt.ylim(0, 10)
    plt.ylabel('MAE')
    imgname1 = 'fig_k_folds_cv_%s.jpg' % name
    plt.savefig(img_dir + '/' + imgname1)
    # LOO score 分布
    imgname2 = None
    imgname3 = None
    if loo:
        # LOO MAE 分布
        plt.figure(figsize=(12, 6))
        maxrow = math.ceil(len(model_names) / 3.0)
        for i in range(len(model_names)):
            plt.subplot(maxrow, 3, i + 1)
            plt.hist(loo_score_list[i], bins=40)
            plt.title(model_names[i])
            plt.xlim(0, 20)
            plt.ylabel('MAE')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        imgname2 = 'fig_loo_dis_%s.jpg' % name
        plt.savefig(img_dir + '/' + imgname2)
        # LOO MAE 概率密度
        plt.figure(figsize=(12, 6))
        x1 = np.arange(0, 20)
        for i in range(len(model_names)):
            plt.subplot(maxrow, 3, i + 1)
            y1 = normfun(x1, df_loo['Mean MAE'][i], np.sqrt(df_loo['Var MAE'][i]))
            plt.plot(x1, y1)
            plt.title(model_names[i])
            plt.xlabel('MAE')
            plt.ylabel('Probability')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        imgname3 = 'fig_loo_norm_%s.jpg' % name
        plt.savefig(img_dir + '/' + imgname3)
    return imgname1, imgname2, imgname3


def plot_metrics(img_dir, name, model_names, df_metrics):
    """
    回归指标图

    :param img_dir:
    :param name:
    :param model_names:
    :param df_metrics:
    :return:
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.bar(np.arange(len(model_names)), df_metrics['MAE'].values, label='MAE')
    for i in range(len(model_names)):
        v = df_metrics['MAE'][i]
        plt.text(i, v + 0.05, '%.4f' % v, ha='center', va='bottom', fontsize=7)
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.title('Mean absolute error for %s' % name)
    plt.ylabel('mean absolute error')
    plt.subplot(122)
    plt.bar(np.arange(len(model_names)), df_metrics['R2'].values, label='R2')
    for i in range(len(model_names)):
        v = df_metrics['R2'][i]
        plt.text(i, v + 0.01, '%.4f' % v, ha='center', va='bottom', fontsize=7)
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.title('R2 for %s' % name)
    plt.ylabel('R2')
    plt.subplots_adjust(bottom=0.2)
    imgname = 'fig_metrics_%s.jpg' % name
    plt.savefig(img_dir + '/' + imgname)
    return imgname


def normfun(x, mu, sigma):
    """
    计算正态分布概率密度

    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    pdf = np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


def main():
    models = create_models()
    inputs = {
        # 'snp_1996': 'tmp/train_1996.csv',
        'snp_119': 'tmp/train_119.csv',
        # 'snp_100': 'tmp/train_100.csv',
        'snp_43': 'tmp/train_43.csv',
        'snp_25': 'tmp/train_25.csv',
        # 'snp_15': 'tmp/train_15.csv',
        'snp_10': 'tmp/train_10.csv',
        # 'snp_5': 'tmp/train_5.csv',
        'snp_1': 'tmp/train_1.csv',
    }
    compare_models(models=models, inputs=inputs, report_dir='report', n_folds=10, loo=False)


if __name__ == '__main__':
    main()
