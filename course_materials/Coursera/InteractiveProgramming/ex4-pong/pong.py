# Implementation of classic arcade game Pong

import simplegui
import random
import math

# initialize globals - pos and vel encode vertical info for paddles
WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH / 2
HALF_PAD_HEIGHT = PAD_HEIGHT / 2
LEFT = False
RIGHT = True

ball_pos = [WIDTH/2, HEIGHT/2]
ball_vel = [1, 1]

paddle1_pos = [PAD_WIDTH/2, HEIGHT/2]
paddle2_pos = [WIDTH-PAD_WIDTH/2, HEIGHT/2]
paddle1_vel = 16
paddle2_vel = 16

score1 = 0
score2 = 0

# initialize ball_pos and ball_vel for new bal in middle of table
# if direction is RIGHT, the ball's velocity is upper right, else upper left
def spawn_ball(direction):
    global ball_pos, ball_vel # these are vectors stored as lists

    # init pos
    ball_pos = [WIDTH/2, HEIGHT/2]

    # init vel
    init_hor_vel = random.randint(2, 4)
    init_ver_vel = random.randint(1, 3)

    if direction == RIGHT:
        ball_vel = [init_hor_vel, -init_ver_vel]
    elif direction == LEFT:
        ball_vel = [-init_hor_vel, -init_ver_vel]

# define event handlers
def new_game():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel  # these are numbers
    global score1, score2  # these are ints

    paddle1_pos = [PAD_WIDTH/2, HEIGHT/2]
    paddle2_pos = [WIDTH-PAD_WIDTH/2, HEIGHT/2]

    spawn_ball(RIGHT)

    score1 = 0
    score2 = 0

def draw(canvas):
    global score1, score2, paddle1_pos, paddle2_pos, ball_pos, ball_vel

    # draw mid line and gutters
    canvas.draw_line([WIDTH / 2, 0],[WIDTH / 2, HEIGHT], 1, "White")
    canvas.draw_line([PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1, "White")
    canvas.draw_line([WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1, "White")

    # update ball
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]

    # vertical velocity
    if ball_pos[1] <= BALL_RADIUS:
        ball_pos[1] = BALL_RADIUS
        ball_vel[1] = -ball_vel[1]
    elif ball_pos[1] >= HEIGHT - BALL_RADIUS:
        ball_pos[1] = HEIGHT - BALL_RADIUS
        ball_vel[1] = -ball_vel[1]

    # horizontal velocity
    if ball_pos[0] <= PAD_WIDTH+BALL_RADIUS:
        ball_pos[0] = PAD_WIDTH+BALL_RADIUS
        # hit paddle1
        if paddle1_pos[1]-HALF_PAD_HEIGHT <= ball_pos[1] <= paddle1_pos[1]+HALF_PAD_HEIGHT:
            ball_vel[0] = -ball_vel[0]
            ########    increase vel by 10%
        else:
            spawn_ball(RIGHT)
            score2 += 1

    elif ball_pos[0] >= WIDTH-PAD_WIDTH-BALL_RADIUS:
        ball_pos[0] = WIDTH-PAD_WIDTH-BALL_RADIUS
        # hit paddle2
        if paddle2_pos[1]-HALF_PAD_HEIGHT <= ball_pos[1] <= paddle2_pos[1]+HALF_PAD_HEIGHT:
            ball_vel[0] = -ball_vel[0]
        else:
            spawn_ball(LEFT)
            score1 += 1

    # draw ball
    canvas.draw_circle(ball_pos, BALL_RADIUS, 1, 'Blue', 'White')

    # draw paddles
    canvas.draw_polygon([(0, paddle1_pos[1]-HALF_PAD_HEIGHT),
                         (0, paddle1_pos[1]+HALF_PAD_HEIGHT),
                         (PAD_WIDTH, paddle1_pos[1]+HALF_PAD_HEIGHT),
                         (PAD_WIDTH, paddle1_pos[1]-HALF_PAD_HEIGHT)], 1, 'Red', 'Red')
    canvas.draw_polygon([(WIDTH-PAD_WIDTH, paddle2_pos[1]-HALF_PAD_HEIGHT),
                         (WIDTH-PAD_WIDTH, paddle2_pos[1]+HALF_PAD_HEIGHT),
                         (WIDTH, paddle2_pos[1]+HALF_PAD_HEIGHT),
                         (WIDTH, paddle2_pos[1]-HALF_PAD_HEIGHT)], 1, 'Blue', 'Blue')
    # draw scores
    canvas.draw_text(str(score1), [100, 100], 36, "White")
    canvas.draw_text(str(score2), [500, 100], 36, "White")

def keydown(key):
    global paddle1_pos, paddle2_pos
    global paddle1_vel, paddle2_vel

    # update paddle1's vertical position
    if key == simplegui.KEY_MAP["s"]:
        paddle1_pos[1] += paddle1_vel
        # keep paddle1 on the screen
        if paddle1_pos[1] >= HEIGHT - HALF_PAD_HEIGHT:
            paddle1_pos[1] = HEIGHT - HALF_PAD_HEIGHT
    # update paddle2's vertical position
    if key == simplegui.KEY_MAP["down"]:
        paddle2_pos[1] += paddle2_vel
        # keep paddle2 on the screen
        if paddle2_pos[1] >= HEIGHT - HALF_PAD_HEIGHT:
            paddle2_pos[1] = HEIGHT - HALF_PAD_HEIGHT

def keyup(key):
    global paddle1_pos, paddle2_pos
    global paddle1_vel, paddle2_vel

    if key == simplegui.KEY_MAP["w"]:
        paddle1_pos[1] -= paddle1_vel
        if paddle1_pos[1] <= HALF_PAD_HEIGHT:
            paddle1_pos[1] = HALF_PAD_HEIGHT
    if key == simplegui.KEY_MAP["up"]:
        paddle2_pos[1] -= paddle2_vel
        if paddle2_pos[1] <= HALF_PAD_HEIGHT:
            paddle2_pos[1] = HALF_PAD_HEIGHT

# create frame
frame = simplegui.create_frame("Pong", WIDTH, HEIGHT)
frame.add_button("Restart", new_game, 200)
frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown)
frame.set_keyup_handler(keyup)


# start frame
new_game()
frame.start()