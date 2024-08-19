import warnings
warnings.filterwarnings("ignore")

from config import collect_args
from executor import Executor
from labeling import Labeler

def main():
    
    args = collect_args()
    print("\n##############################")

    stop = 0
    while stop == 0:
        executor = Executor(args)
        stop = executor.execute_mode()
        print("\n##############################")
    
    labeler = Labeler(args)
    labeler.run()

if __name__ == "__main__":
    main()