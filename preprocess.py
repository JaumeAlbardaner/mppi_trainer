import pandas as pd
import sys
from matplotlib import pyplot as plt
import numpy as np

def usage():
  print("USAGE: python3 preprocess.py <input.csv>")

  
  
def main(file_csv):
    train = pd.read_csv(file_csv)
    x = train[["roll", "u_x","u_y", "yaw_der", "steering", "throttle"]].to_numpy()
    y = np.empty([len(x), 4], dtype=np.float64)

    #Preprocessing
    print("Preprocessing . . .")
    print("Building response . . .")
    y[:,0] = np.multiply(np.cos(train["yaw"]),train["u_x"]) - np.multiply(np.sin(train["yaw"]),train["u_y"])
    y[:,1] = np.multiply(np.sin(train["yaw"]),train["u_x"]) + np.multiply(np.cos(train["yaw"]),train["u_y"])
    y[:,2] = -train["yaw_der"]
    y[:,3] = train["roll_der"]


    print("Filtering results . . .")

    for i in range(len(x)):
        #Roll
        if abs(x[i,0]) > 0.2:
            x[i,0] = x[i-1,0]
        #u_x
        if abs(x[i,1]) > 3 or x[i,1] < 0:
            x[i,1] = x[i-1,1]
        #u_y
        if abs(x[i,2]) > 1.5:
            x[i,2] = x[i-1,2]
        #yaw_der
        if abs(x[i,3]) > 1.5:
            x[i,3] = x[i-1,3]

    for i in range(len(y)):
        #u_x_wlrd
        if abs(y[i,0]) > 3:
            y[i,0] = y[i-1,0]
        #u_y_wlrd
        if abs(y[i,1]) > 3:
            y[i,1] = y[i-1,1]
        # - yaw_der
        if abs(y[i,2]) > 1.5:
            y[i,2] = y[i-1,2]
        #Roll_der
        if abs(y[i,3]) > 0.3:
            y[i,3] = y[i-1,3]

    df = pd.DataFrame({"roll" : x[:,0], "u_x" : x[:,1], "u_y" : x[:,2], "yaw_der" : x[:,3], "steering" : x[:,4], "throttle" : x[:,5],
     "u_x_der" : y[:,0], "u_y_der" : y[:,1], "yaw_der_der" : y[:,2], "roll_der" : y[:,3]})
    df.to_csv(file_csv[:-4] +"_preprocessed.csv", index=False)

    x_axis = list(range(len(x)))
    
    fig, axs = plt.subplots(8)
    fig.suptitle('Preprocessed inputs')
    axs[0].plot(x_axis,x[:,0], color='tab:blue',    label = "roll"    )
    axs[0].legend()
    axs[1].plot(x_axis,x[:,1], color='tab:orange',  label = "u_x"     )
    axs[1].legend()
    axs[2].plot(x_axis,x[:,2], color='tab:red',     label = "u_y"     )
    axs[2].legend()
    axs[3].plot(x_axis,x[:,3], color='tab:gray',   label = "yaw_der"  )
    axs[3].legend()

    axs[4].plot(x_axis,y[:,0], color='tab:blue',    label = "u_x_der"    )
    axs[4].legend()
    axs[5].plot(x_axis,y[:,1], color='tab:orange',  label = "u_y_der"     )
    axs[5].legend()
    axs[6].plot(x_axis,y[:,2], color='tab:red',     label = "yaw_der_der"     )
    axs[6].legend()
    axs[7].plot(x_axis,y[:,3], color='tab:gray',   label = "roll_der"  )
    axs[7].legend()
    plt.savefig('Preprocessed_inputs_'+file_csv[:-4] +'.pdf')



if __name__ == '__main__':
  if len(sys.argv) != 2:
    usage()
  else:
    csv_file = sys.argv[1]
    main(csv_file)