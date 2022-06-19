import pandas as pd
import sys
from matplotlib import pyplot as plt
import numpy as np

def usage():
  print("USAGE: python3 preprocess.py <file.csv>")

  
  
def main(file_csv):
    train = pd.read_csv(file_csv, dtype=float)
    # {"Input0": np.float64,
    #                                     "Input1": np.float64, 
    #                                     "Input2": np.float64,
    #                                     "Input3": np.float64,
    #                                     "Control0": np.float64,
    #                                     "Control1": np.float64,
    #                                     "Der0": np.float64,
    #                                     "Der1": np.float64,
    #                                     "Der2": np.float64,
    #                                     "Der3": np.float64,
    #                                     "dummy": np.int8})
    x = train[["Control0", "Control1"]].to_numpy()
    # y = train[["Der0", "Der1", "Der2", "Der3"]].to_numpy()

    # df = pd.DataFrame({"roll" : x[:,0], "u_x" : x[:,1], "u_y" : x[:,2], "yaw_der" : x[:,3], "steering" : x[:,4], "throttle" : x[:,5],
    #  "roll_der" : y[:,0], "u_x_der" : y[:,1], "u_y_der" : y[:,2], "yaw_der_der" : y[:,3]})
    # df.to_csv(file_csv[:-4] +"_preprocessed.csv", index=False)

    x_axis = list(range(len(x)))
    
    fig, axs = plt.subplots(10)
    fig.suptitle('Preprocessed Dynamic')
    axs[0].plot(x_axis,x[:,0], color='tab:blue',    label = "Input 0"    )
    axs[0].legend()
    axs[1].plot(x_axis,x[:,1], color='tab:orange',  label = "Input 1"     )
    axs[1].legend()
    # axs[2].plot(x_axis,x[:,2], color='tab:red',     label = "Input 2"     )
    # axs[2].legend()
    # axs[3].plot(x_axis,x[:,3], color='tab:gray',   label = "Input 3"  )
    # axs[3].legend()
    # axs[4].plot(x_axis,x[:,4], color='tab:purple',   label = "Control 0"  )
    # axs[4].legend()
    # axs[5].plot(x_axis,x[:,5], color='tab:green',   label = "Control 1"  )
    # axs[5].legend()


    # axs[6].plot(x_axis,y[:,0], color='tab:blue',    label = "Output 0"    )
    # axs[6].legend()
    # axs[7].plot(x_axis,y[:,1], color='tab:orange',  label = "Output 1"     )
    # axs[7].legend()
    # axs[8].plot(x_axis,y[:,2], color='tab:red',     label = "Output 2"     )
    # axs[8].legend()
    # axs[9].plot(x_axis,y[:,3], color='tab:gray',   label = "Output 3"  )
    # axs[9].legend()
    plt.savefig('Visualized_Dynamic_'+file_csv[:-4] +'data.pdf')



if __name__ == '__main__':
  if len(sys.argv) != 2:
    usage()
  else:
    csv_file = sys.argv[1]
    main(csv_file)