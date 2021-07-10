from tkinter import *
import io
from PIL import Image
from digits import get_results
from os import remove
from os.path import isfile


class GUI:
    def __init__(self, master):  # Initialize vars
        self.master = master
        self.old_x = None
        self.old_y = None
        self.penwidth = 30
        self.drawWidgets()  # Draw controls: canvas & buttons & labels
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.master.protocol("WM_DELETE_WINDOW", self.close_window)

    def paint(self, e):  # Paint on canvas
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill='black',
                               capstyle=ROUND, smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # Stop painting
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.c.delete(ALL)
        self.update_labels(clear=True)

    def update_labels(self, num=0, acc=0, clear=False):  # Change labels (or clear them)
        if clear:
            self.num_label['text'] = ''
            self.acc_label['text'] = ''
        else:
            self.num_label['text'] = f'Number: {num}'
            self.acc_label['text'] = f'Accuracy: {acc}%'

    def submit_img(self):  # Send image to model and get a prediction
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('tmp_img.jpg')
        index, val = get_results('tmp_img.jpg')
        if index is None or val is None:
            self.update_labels(clear=True)
        else:
            self.update_labels(index, val)

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        self.submit_btn = Button(self.controls, text='Submit Image', font='arial 18', command=self.submit_img)
        self.clear_btn = Button(self.controls, text='Clear Canvas', font='arial 18', command=self.clear)

        self.submit_btn.grid(row=0, column=0)
        self.clear_btn.grid(row=1, column=0)

        self.num_label = Label(self.controls, text='', font=('arial 12'))
        self.acc_label = Label(self.controls, text='', font=('arial 12'))

        self.num_label.grid(row=2, column=0)
        self.acc_label.grid(row=3, column=0)

        self.controls.pack(side=LEFT)

        self.c = Canvas(self.master, width=800, height=400, bg='white')
        self.c.pack(fill=BOTH, expand=True)

    def close_window(self):  # Delete temp image before closing app
        if isfile('tmp_img.jpg'):
            remove('tmp_img.jpg')
        self.master.destroy()


def main():
    root = Tk()
    GUI(root)
    root.title('Tensorflow Digit Recognition')
    root.mainloop()


if __name__ == '__main__':
    main()
