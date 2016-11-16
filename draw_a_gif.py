import sys
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(4,4))
fig.set_tight_layout(True)
#  询问图形在屏幕上的尺寸和DPI（每英寸点数）。
#  注意当我们把图形储存成一个文件时，我们需要再另外提供一个DPI值
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

inpu_LIST=[[1,1,1,1]]
goal=[1.82546,40.329,1.250,0.900]
print(len(inpu_LIST))

tmp = np.loadtxt("data.csv", dtype="double", delimiter=",")
dataset = tmp.tolist()
datax = dataset[0]
last_frame=0


def showPic(ll=[],label=""):
    a1,a2,r1,r2=ll#[-5.004003806672138, 2.954498879245496, -1.3446282102078375, 1.1512684349546953]#[-4.404003806672138, 1.7544988792454959, -3.144628210207838, 0.5512684349546954]#[2.1959961933278613, 17.954498879245495, -15.144628210207832, -33.64873156504534]#[-5.004003806672138, 2.954498879245496, -1.3446282102078375, 1.1512684349546953]#[-2.6040038066721385, 0.5544988792454958, 2.8553717897921627, -1.248731565045305]
    formulation = lambda x: ((a2 * ((r2 - x) ** 3)) + (a1 * ((r1 - x) ** 3))) * (x < r2) + a1 * ((r1 - x) ** 3) * (
        x > r2 and x < r1)
    datay=list(map(formulation,datax))
    a1,a2,r1,r2=goal
    goaly=list(map(formulation,datax))
    plt.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(label)
    plt.scatter(datax,goaly,color='red')
    plt.scatter(datax,datay,color='blue')
    plt.axis([-0.2,1.6,-5,50])
   # plt.title(title)
    fig.suptitle('DQN')
    return ax
def update(i=0):
    global last_frame
    if i >= last_frame:
        i=last_frame
    label = 'timestep {0}'.format(i)
    return showPic(inpu_LIST[i],label)

def save_the_pic(path,inpu_list):
    global last_frame
    last_frame = len(inpu_LIST)
    global inpu_LIST
    inpu_LIST=inpu_list
    print(inpu_LIST)
    anim = FuncAnimation(fig, update, frames=np.arange(0,len(inpu_LIST)+10), interval=200) #+10为了加长最后一幁的时间
    anim.save(path+'/'+ "my_exp.gif", dpi=80, writer='imagemagick')


if __name__ == '__main__':
    # FuncAnimation 会在每一帧都调用“update” 函数。
    # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
    title="DQN"
    anim = FuncAnimation(fig, update, frames=np.arange(0,len(inpu_LIST)), interval=200)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save(title+".gif", dpi=80, writer='imagemagick')
    else:
     #    plt.show() 会一直循环播放动画
        plt.show()