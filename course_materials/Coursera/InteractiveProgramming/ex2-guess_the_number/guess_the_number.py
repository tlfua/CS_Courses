# template for "Guess the number" mini-project
# input will come from buttons and an input field
# all output for the game will be printed in the console

import simplegui
import random
import math


range_max = 100
remain_guesses = 0
secret_number = 0

i_guess = 0

# helper function to start and restart the game
def new_game():
    # initialize global variables used in your code here
    global range_max
    global remain_guesses
    global secret_number

    print "\nNew game. Range is from 0 to ", range_max

    if range_max == 100:
        remain_guesses = int(math.ceil(math.log(100, 2)))
        print "Number of remaining guesses is ", remain_guesses
        secret_number = random.randrange(0, 100)
    elif range_max == 1000:
        remain_guesses = int(math.ceil(math.log(1000, 2)))
        print "Number of remaining guesses is ", remain_guesses
        secret_number = random.randrange(0, 1000)

# define event handlers for control panel
def range100():
    # button that changes the range to [0,100) and starts a new game
    global range_max
    range_max = 100
    new_game()

def range1000():
    # button that changes the range to [0,1000) and starts a new game
    global range_max
    range_max = 1000
    new_game()

def input_guess(guess):
    # main game logic goes here
    global i_guess
    global remain_guesses
    global secret_number

    i_guess = int(guess)
    print "\nGuess was ", i_guess

    # print comparison result
    if i_guess > secret_number:
        print "Higher!"

        remain_guesses = remain_guesses - 1
        print "Number of remaining guesses is ", remain_guesses

        if remain_guesses == 0:
            print "You ran out of guesses, The number was ", secret_number
            new_game()

    elif i_guess < secret_number:
        print "Lower!"

        remain_guesses = remain_guesses - 1
        print "Number of remaining guesses is ", remain_guesses

        if remain_guesses == 0:
            print "You ran out of guesses, The number was ", secret_number
            new_game()

    elif i_guess == secret_number:
        print "Correct!"

        remain_guesses = remain_guesses - 1
        print "Number of remaining guesses is ", remain_guesses

        new_game()

# create frame
f = simplegui.create_frame("Guess the number", 200, 200)

# control elements for frame
f.add_button("Range is [0, 100)", range100, 200)
f.add_button("Range is [0, 1000)", range1000, 200)
f.add_input("Enter a guess", input_guess, 200)

new_game()

f.start()