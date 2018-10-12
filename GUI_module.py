import cv2
from tkinter import *
import tkinter.messagebox
from GUI.widgets import *
from PIL import Image, ImageTk
# from utils.emotion_recognize import *
from utils.video_process import *
from tkinter import ttk


class Window:
    def __init__(self):
        self.window = Tk()
        self.window.title('人脸日记')
        self.window.size()
        sw = self.window.winfo_screenwidth()  # 获取屏幕宽
        sh = self.window.winfo_screenheight()  # 获取屏幕高
        wx = 850
        wh = 600
        self.window.geometry("%dx%d+%d+%d" %(wx, wh, (sw-wx)/2, (sh-wh)/2-100))  # 窗口至指定位置
        self.videoProc = VideoProcessor(self)
        # ---------------------------控件
        self.bLogin = Button(self.window, text='登陆', width=10, command=self.loginButtonFuntion)
        self.bLogin.place(x=745, y=200)
        self.bExit = Button(self.window, text='退出', width=10, command=self.exit)
        self.bExit.place(x=745, y=480)
        self.canvas = ICanvas(self.window, bg='#ffffff', height=480, width=720)
        self.canvas.place(x=0, y=120)
        self.bAbout = Button(self.window, text='关于', width=10,
                             command=lambda: tkinter.messagebox.showinfo(title='关于',
                             message='由bo233及其团队编写的人脸“日记”。\n\n由于时间和技术水平有限，功能可能不够完善，敬请谅解。\n\n'
                                     '感谢CSDN、GitHub、StackOverflow，感谢为开源项目奋斗的前辈们，'
                                     '是你们的奉献使得该程序的完成成为可能。'))
        self.bAbout.place(x=745, y=410)
        self.bLogon = Button(self.window, text='注册', width=10,
                             command=lambda: self.logonButtonFunction())
        self.bLogon.place(x=745, y=270)
        self.bHelp = Button(self.window, text='帮助', width=10)
        self.bHelp.place(x=745, y=340)

        self.bLogout = Button(self.window, text='登出', width=10, command=self.logoutButtonFuntion)
        self.bRecordEmotion = Button(self.window, text='记录心情', width=10, command=self.emotionButtonFunction)
        self.bReadDiary = Button(self.window, text='查看日记', width=10, command=self.readDiaryButtonFunction)

        self.window.resizable(0, 0)
        # 刚进入界面时进行表情识别
        self.videoProc.runControl()
        self.window.mainloop()

    def exit(self):
        self.videoProc.runCommand = -1
        self.window.destroy()
        self.videoProc.camera.release()

    def logoutButtonFuntion(self):
        tkinter.messagebox.showinfo(title='登出', message='感谢使用，欢迎再次使用！')
        self.bLogin.place(x=745, y=200)
        self.bLogon.place(x=745, y=270)
        self.bLogout.place_forget()
        self.bReadDiary.place_forget()
        self.bRecordEmotion.place_forget()
        self.lWelcome.place_forget()

    # 表情按钮相关执行函数，包括表情识别，表情确定，记录表情
    def emotionButtonFunction(self):
        self.videoProc.run(self.videoProc.RUN_EMOTION_RECOG, record=True)
        yes = tkinter.messagebox.askyesno(title='记录心情', message='识别出你的表情为'+
                            self.videoProc.emotionLabels[np.argmax(self.videoProc.emotionRecord)]+
                                    '，是否记录表情？')
        if yes:
            fo = open('./data/at/%s/diary.txt' % str(self.videoProc.whoRU), 'a')
            fo.write(str(time.strftime('\n%Y-%m-%d %H:%M', time.localtime()))+'    '
                     +self.videoProc.emotionLabels[np.argmax(self.videoProc.emotionRecord)])
            fo.close()


    # 登陆按钮相关函数，包括人脸识别并且匹配到相应用户
    def loginButtonFuntion(self):
        self.bLogon.config(state='disabled')
        self.bLogin.config(state='disabled')
        self.videoProc.run(self.videoProc.RUN_FACE_RECOG)
        if self.videoProc.whoRU != -1:
            tkinter.messagebox.showinfo(title='登陆成功', message=self.videoProc.names[self.videoProc.whoRU]+'，欢迎你！')
            self.bLogin.place_forget()
            self.bLogon.place_forget()
            self.bLogout.place(x=745, y=270)
            self.bReadDiary.place(x=745, y=200)
            self.bRecordEmotion.place(x=745, y=130)
            self.lWelcome = tk.Label(self.window, text=self.videoProc.names[self.videoProc.whoRU]+'\n欢迎你')
            self.lWelcome.config(font='Blackletter -20')
            self.lWelcome.place(x=750, y=30)
        else:
            tkinter.messagebox.showinfo(title='登陆失败', message='抱歉，无法识别你！')
        self.bLogon.config(state='normal')
        self.bLogin.config(state='normal')

    # 注册按钮相关函数
    def logonButtonFunction(self):
        self.bLogon.config(state='disabled')
        self.bLogin.config(state='disabled')

        class TextMsgbox:
            def __init__(self):
                self.name = ''
                self.root = tk.Tk()
                sw = self.root.winfo_screenwidth()  # 获取屏幕宽
                sh = self.root.winfo_screenheight()  # 获取屏幕高
                self.root.title("注册")
                self.root.geometry("%dx%d+%d+%d" % (300, 80, (sw - 300) / 2, (sh - 80) / 2 - 100))
                self.l1 = tk.Label(self.root, text="请输入姓名：")
                self.l1.pack()
                self.xls = tk.Entry(self.root)
                self.xls.pack()
                self.button = Button(self.root, text="确认", width=7, command=self.getName)
                self.button.pack()
                # ##########可以的话加个取消
                self.root.mainloop()

            def getName(self):
                self.name = self.xls.get()
                self.root.quit()
                self.root.destroy()

        # 弹框提示用户输入姓名
        textBox = TextMsgbox()
        name = textBox.name
        path = './data/at/'+str(self.videoProc.faceRegistered)
        if name == '':
            tkinter.messagebox.showinfo(title='提示', message='请输入姓名！')
        else:
            os.mkdir(path)
            self.videoProc.run(self.videoProc.RUN_FACE_GENERATE, name=name)
            tkinter.messagebox.showinfo(title='提示', message='注册成功！')
        self.bLogon.config(state='normal')
        self.bLogin.config(state='normal')

    def readDiaryButtonFunction(self):
        class DiaryView:
            def __init__(self, num):
                path = './data/at/' + str(num) + '/diary.txt'
                of = open(path)
                fread = of.readlines()
                self.root = tkinter.Tk()
                self.root.title('查看日记')
                self.tree = ttk.Treeview(self.root, height=20, show="headings")  # 表格
                self.tree['columns'] = ('日期', '心情')
                # 表示列,不显示
                self.tree.column('日期', width=200, anchor='center')
                self.tree.column('心情', width=100, anchor='center')
                self.tree.heading('日期', text="日期")  # 显示表头
                self.tree.heading('心情', text="心情")
                i = 0
                for raw in fread:
                    self.tree.insert('', i, values=(raw[0:17], raw[18:]))
                    i += 1
                self.tree.pack()
                of.close()
                self.root.protocol("WM_DELETE_WINDOW", self.exit)
                self.root.mainloop()
            def exit(self):
                self.root.destroy()
                self.root.quit()

        DiaryView(self.videoProc.whoRU)


if __name__ == '__main__':
    w = Window()
