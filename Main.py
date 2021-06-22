from Metaheuristics import *
from problem.Problem import*
from evaluate.Evaluate import *

problem=Problem("functions")
evaluate=Evaluate()
parameters=[{"to":1000,"ta":0.001,"delta":0.99}]
x=Simulated.Simulated((4,2))#,parameters)
y=Simulated.Simulated((1,2))
z=Fwa.Fwa((20,2))
aa=Pso.Pso((5,2))
bb=Mine.Mine((5,2))
metas = [x,y,z,aa,bb]
evaluate.eva(metas,problem,5,visual=True)
#solution, fitness=x.simulated(problem)
#solution2, fitness2=y.simulated(problem)
#print(solution, fitness)
#print(solution2, fitness2)



"""
from tkinter import *
def calculate():
    temp = int(entry.get())
    temp = 9/5*temp+32
    output_label.configure(text = 'Converted: {:.1f}'.format(temp))
    entry.delete(0,END)
root = Tk()
message_label = Label(text='Enter a temperature',
font=('Verdana', 16))
output_label = Label(font=('Verdana', 16))
entry = Entry(font=('Verdana', 16), width=4)
calc_button = Button(text='Ok', font=('Verdana', 16),
command=calculate)
message_label.grid(row=0, column=0)
entry.grid(row=0, column=1)
calc_button.grid(row=0, column=2)
output_label.grid(row=1, column=0, columnspan=3)
mainloop()
"""
