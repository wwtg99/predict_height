import preprocess
import argparse
import models


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help='Input features csv file')
    parser.add_argument('-m', '--model', help='Predict model, comma to separate', default='gbr')
    parser.add_argument('-d', '--model-dir', help='Model directory', default='models')
    parser.add_argument('-l', '--list-models', help='List available models', action='store_true')
    parser.add_argument('-q', '--quiet', help='Only show predict result and errors', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = init_args()
    if args.list_models:
        print('\n'.join(models.get_model_names()))
        exit()
    m = args.model.split(',')
    x, y = preprocess.load_data(args.input_csv)
    for model_name in m:
        model = models.load_model(model_name, args.model_dir)
        if model is None:
            if not args.quiet:
                print('Invalid model %s' % model_name)
            continue
        if not args.quiet:
            print('Load model %s' % model_name)
        pre_y = model.predict(x)
        if not args.quiet:
            print('Predict by model %s: %.2f' % (model_name, pre_y[0]))
        else:
            print('%.2f' % pre_y[0])


if __name__ == '__main__':
    main()
