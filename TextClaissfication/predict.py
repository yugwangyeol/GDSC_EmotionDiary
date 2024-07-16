import argparse

from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = 'Output_elice50/model_out'

predictor = BertClassificationPredictor(
    model_path=MODEL_PATH,
    label_path='Data3/label',  # location for labels.csv file
    multi_label=False,
    model_type='bert',
    do_lower_case=False,
    device=None)  # set custom torch.device, defaults to cuda if available

def text_generator(file_root):
    text = []
    f = open(file_root, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip() # 줄 끝의 줄 바꿈 문자를 제거한다.
        text.append(line)
    f.close()
    str = ''.join(text)
    return str

def write_txt(decision,args):
    f = open(f'test/{args.user_id}/diary/{args.diary_id}/output.txt','w')
    f.write(decision)
    f.close()

def get_args():
    parser = argparse.ArgumentParser(
        description='Predict depression from input text')
    parser.add_argument('--user_id', required=True, type=str,
                        help='Please enter analysis text')
    parser.add_argument('--diary_id', required=True, type=str,
                        help='Please enter analysis text')
    return parser.parse_args()

def text_classification(args):
    file_root = f'test/{args.user_id}/diary/{args.diary_id}/content.txt'
    text = text_generator(file_root)
    single_prediction = predictor.predict(text)
    if single_prediction[0][0] == '1':
        return write_txt('depression',args)
    else:
        return write_txt('nondepression',args)


if __name__ == "__main__":
    args = get_args()
    text_classification(args)
    
