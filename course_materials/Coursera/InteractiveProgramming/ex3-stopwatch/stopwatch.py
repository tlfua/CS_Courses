# template for "Stopwatch: The Game"

import simplegui
import math

# define global variables
is_counting = False
count_time = 0
str_time = "0:00.0"

scores = [0, 0]

# define helper function format that converts time
# in tenths of seconds into formatted string A:BC.D
def format(t):

    BC = t // 10
    D = t % 10
    A = BC // 60
    BC = BC % 60
    # print A, BC, D

    if BC // 10 == 0:
        return str(A) + ":0" + str(BC) + "." + str(D)
    else:
        return str(A) + ":" + str(BC) + "." + str(D)

# format(0)
# format(11)
# format(321)
# format(613)

# define event handlers for buttons; "Start", "Stop", "Reset"
def Start():
    global is_counting
    is_counting = True

def Stop():
    global is_counting
    global str_time

    if is_counting == True:

        if str_time[-1] == "0":
            scores[0] = scores[0] + 1
        scores[1] = scores[1] + 1

    is_counting = False

def Reset():
    global is_counting
    global count_time
    global str_time

    is_counting = False
    count_time = 0
    str_time = format(count_time)

    global scores
    scores = [0, 0]

# define event handler for timer with 0.1 sec interval
def tick():
    global is_counting
    global count_time
    global str_time

    # print count_time
    if is_counting:
        count_time = count_time + 1
        str_time = format(count_time)

# define draw handler
def draw(canvas):
    canvas.draw_text(str_time, [100, 100], 36, "White")
    canvas.draw_text(str(scores[0]) + "/" + str(scores[1])
                             , [250, 20], 24, "Green")

# create frame
frame = simplegui.create_frame("Stopwatch", 300, 200)

# register event handlers
frame.set_draw_handler(draw)
timer = simplegui.create_timer(100, tick)

frame.add_button("Start", Start, 200)
frame.add_button("Stop", Stop, 200)
frame.add_button("Reset", Reset, 200)

# start frame
frame.start()
timer.start()

# Please remember to review the grading rubric