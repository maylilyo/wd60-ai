# Standard
from pathlib import Path

# PIP
from ptflops import get_model_complexity_info

# Custom
from config import Config
from softcode.main_net import Main_net


MODEL_ID = '1634091753'  # pwc-net

PROJECT_DIR = Path(__file__).absolute().parent
OUTPUT_DIR = PROJECT_DIR / 'output'


def main():
    cfg = Config()
    cfg.load_options(MODEL_ID)
    model = Main_net(cfg.model_option)
    model.cuda()

    macs, params = get_model_complexity_info(model, (3, 256, 448))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
