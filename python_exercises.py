# 7.1 Scripting Exercises
'''
1. Write a Python script which computes the gravitational force between (a) the
Earth and the Moon, and (b) The Sun and the Moon (you will need to look
up masses and mean distances). The code should print the result as a number
formatted to two decimal places. This last step involves looking up how to
output formatted numbers in Python.
'''
earth_mass = 5.972 * (10**24)
moon_mass = 7.34767309 * (10**22)
sun_mass = 1.989 * (10**30)
distance_earth_moon = 3.844 * (10**8)
distance_sun_moon = 1.5 * (10**11)
gravitational_constant = 6.6743 * (10**-11)
gravity_earth_moon = (gravitational_constant * earth_mass * moon_mass) / (distance_earth_moon ** 2)
gravity_sun_moon = (gravitational_constant * sun_mass * moon_mass) / (distance_sun_moon ** 2)

# print(f"{distance_earth_moon:.2e}")
# print(f"{distance_sun_moon:.2e}")
print("PROBLEM 1: Gravitational force")
print(f"The mass of the earth is {earth_mass:.2e} kg.\nThe mass of the moon is {moon_mass:.2e} kg.")
print(f"The distance between the Earth and the Moon is {distance_earth_moon:.2e} km.\nThe distance between the Sun and the Moon is {distance_sun_moon:.2e} km.")
print(f"The gravitational constant is {gravitational_constant:.2e} m^3 kg^-1 s^-2.")
print(f"The gravitational force between the Earth and the Moon is {gravity_earth_moon:.2e}.\nThe gravitational force between the Sun and the Moon is {gravity_sun_moon:.2e}.")

# Output:
# PROBLEM 1: Gravitational force
# The mass of the earth is 5.97e+24 kg.
# The mass of the moon is 7.35e+22 kg.
# The distance between the Earth and the Moon is 3.84e+08 km.
# The distance between the Sun and the Moon is 1.50e+11 km.
# The gravitational constant is 6.67e-11 m^3 kg^-1 s^-2.
# The gravitational force between the Earth and the Moon is 1.98e+20.
# The gravitational force between the Sun and the Moon is 4.34e+20.

'''
2. Write a Python script which defines the electric charges of two point sources as
variables (feel free to pick the charges), and the separation between them, then
computes the electrostatic force between them.
'''
q1_charge = 3 * (10**-6)
q2_charge = 1.50 * (10**-6)
distance_q1_q2 = 0.12
q1_q2_charge_value = 4.5 * (10**-12)
coulomb_constant = 8.99 * (10**9)
electrostatic_force = ((coulomb_constant) * (q1_q2_charge_value)) / (distance_q1_q2 ** 2) # Coulomb's law

print(f"\nPROBLEM 2: Electrostatic force")
print(f"The charge of the first point is {q1_charge:.2e} C.\nThe charge of the second point is {q2_charge:.2e} C.")
print(f"The distance between the first and second points is {distance_q1_q2:.3} m.")
print(f"The electrostatic force between the two points is {electrostatic_force:.3} N.")

# Output:
# PROBLEM 2: Electrostatic force
# The charge of the first point is 3.00e-06 C.
# The charge of the second point is 1.50e-06 C.
# The distance between the first and second points is 0.12 m.
# The electrostatic force between the two points is 2.81 N.

'''
3. Write a Python script to compute the distance between the (user inputted) 
points (x1, y1) and (x2, y2) on a Euclidean plane.
'''
print(f"\nPROBLEM 3: Distance between coordinates")

x1, y1 = float(input("Coordinate x1: ")), float(input("Coordinate y1: "))
x2, y2 = float(input("Coordinate x2: ")), float(input("Coordinate y2: "))
distance_inputs = (((x2-x1) ** 2) + ((y2-y1) ** 2)) ** 0.5

print(f"Distance between coordinates: {distance_inputs:.2f}")

# Output :
# PROBLEM 3: Distance between coordinates
# Coordinate x1: 1 
# Coordinate y1: 2
# Coordinate x2: 3
# Coordinate y2: 4
# Distance between coordinates: 2.83

# 7.2. Array Exercises
'''
1. Make a list containing the numbers [1, 2, 4, 7, 3, 6, 91, 2001, 42]. 
Then, print the first and last entries of the resulting list to the screen.
'''
numbers = [1, 2, 4, 7, 3, 6, 91, 2001, 42]

print("\n7.2 Exercises\nPROBLEM 1: First & Last elements\nList of numbers: 1, 2, 4, 7, 3, 6, 91, 2001, 42")
print(f"First entry: {numbers[0]}")
print(f"Last entry: {numbers[-1]}")

# Output: 
# 7.2 Exercises
# PROBLEM 1: First & Last elements
# List of numbers: 1, 2, 4, 7, 3, 6, 91, 2001, 42
# First entry: 1
# Last entry: 42

'''
2. Make a list containing the strings [Red, Green, White, Black]. 
Then, add fuschia and aquamarine to the BEGINNING of the list of strings 
created just above.
'''
# the .insert method isn't mentioned in the Python exercises doc, but I found .insert() works basically the same as .splice() in JS (of which I have familiarity)
print("\nPROBLEM 2: Add fuschia and aquamarine\nList of strings: Red, Green, White, Black")
list_of_colors = ["Red", "Green", "White", "Black"]
list_of_colors.insert(0, "fuschia")
list_of_colors.insert(0, "aquamarine")
print(list_of_colors)

# Output:
# PROBLEM 2: Add fuschia and aquamarine
# List of strings: Red, Green, White, Black
# ['aquamarine', 'fuschia', 'Red', 'Green', 'White', 'Black']

'''
3. Make a list from [4, 23, 4, 6, 8, 4, 3, 87, 9, 4, 3, 6, 7, 5, 3, 2, 4, 33, 5, 3]. 
Then, write some code that counts the number of times the number 4 appears 
in the list and stores that count in a new variable.
'''
# again, the .count() method isn't mentioned in the doc so I Googled the correct method to use
print("\nPROBLEM 3: Counting 4\nList of numbers: 4, 23, 4, 6, 8, 4, 3, 87, 9, 4, 3, 6, 7, 5, 3, 2, 4, 33, 5, 3")
list_of_numbers = [4, 23, 4, 6, 8, 4, 3, 87, 9, 4, 3, 6, 7, 5, 3, 2, 4, 33, 5, 3]
count_4 = list_of_numbers.count(4)
print(f"Number of 4s in list of numbers: {count_4}")

# Output: 
# PROBLEM 3: Counting 4
# List of numbers: 4, 23, 4, 6, 8, 4, 3, 87, 9, 4, 3, 6, 7, 5, 3, 2, 4, 33, 5, 3
# Number of 4s in list of numbers: 5

'''
4. Take the list of numbers from above, cut it in half, so that you have two lists, 
then make a new list with two entries, each containing one of these lists.
'''
print("\nPROBLEM 4: Cut and combine")
half_list = len(list_of_numbers) // 2
first_half = list_of_numbers[:half_list]
second_half = list_of_numbers[half_list:]
combined_list = [first_half, second_half]
print("Cut and combined list:", *combined_list)

# Output: 
# PROBLEM 4: Cut and combine
# Cut and combined list: [4, 23, 4, 6, 8, 4, 3, 87, 9, 4] [3, 6, 7, 5, 3, 2, 4, 33, 5, 3]

'''
5. Write code that takes two inputs - a string and an integer. 
Then, generate a list with the integer number of elements, each containing 
the string.
'''
print("\nPROBLEM 5: Integer long list of 'string'")
string = input("String: ")
integer = int(input("Integer: "))
answer_list = [string] * integer
print("List:", *answer_list)

# Output: 
# PROBLEM 5: Integer long list of 'string'
# String: string
# Integer: 3
# List: string string string

'''INTERMEDIARY'''
print("\nThe following problems do not accept inputs and print very long lists of whatever.\nPrepare yourself.")

yes = input("Enter 'yes' to continue: ")

if yes == "yes":
    print("Continuing.")
else:
    print("Continuing anyways!")

'''
6. Generate a three element list, each element of which is itself a three 
element list, each element of which is ITSELF a three element list, and then 
make all those elements contain the string 'rabbit'.
'''
print("\nPROBLEM 6: Three element list in a list in a list; rabbit")
full_list = []
for a in range(3):
    first_list = []
    for b in range(3):
        second_list = []
        for c in range(3):
            second_list.append("rabbit")
        first_list.append(second_list)
    full_list.append(first_list)

print(full_list)

# Output: 
# PROBLEM 6: Three element list in a list in a list; rabbit
# [[['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit']], [['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit']], [['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit'], ['rabbit', 'rabbit', 'rabbit']]]

'''
7. Create a list from 2, 5, 8, 2, 6, 4, 3, and then converts that list into 
a single integer that’s the list elements all multiplied together.
'''
print("\nPROBLEM 7: List elements multiplied")
list_of_numbers = [2, 5, 8, 2, 6, 4, 3]
multiplied = 1
for number in list_of_numbers:
    multiplied *= number
print("List of numbers: 2, 5, 8, 2, 6, 4, 3")
print(f"All elements multiplied: {multiplied}")

# Output: 
# PROBLEM 7: List elements multiplied
# List of numbers: 2, 5, 8, 2, 6, 4, 3
# All elements multiplied: 11520

'''
8. Same as above, except this time take the list elements and CONCATENATE them 
together into a single string, then turn that string into an integer. 
At the end, you should have the integer 2582643.
'''
print("\nPROBLEM 8: Concatenate list elements into string, then string to integer")
string_numbers = ""
for num in list_of_numbers:
    string_numbers += str(num)
integer_answer = int(string_numbers)
print(f"Concatenated integer: {integer_answer}")

# Output: 
# PROBLEM 8: Concatenate list elements into string, then string to integer
# Concatenated integer: 2582643

'''
9. Take 11, 43, 52 and "bunny", "cat", "pony" and make a dict using the former 
as the keys.
'''
print("\nPROBLEM 9: Making a dictionary\nKeys: 11, 43, 52\nDictionary values: bunny, cat, pony")
keys = [11, 43, 52]
dict_words = ["bunny", "cat", "pony"]
my_dict = dict(zip(keys, dict_words))
print("Dictionary: ", my_dict)

# Output: 
# PROBLEM 9: Making a dictionary
# Keys: 11, 43, 52
# Dictionary values: bunny, cat, pony
# Dictionary:  {11: 'bunny', 43: 'cat', 52: 'pony'}

'''
10. Add 64, as key to "puppy", to the above dict.
'''
print("\nPROBLEM 10: Adding puppy and 64")
my_dict[64] = "puppy"
print("Updated dictionary: ", my_dict)

# Output: 
# PROBLEM 10: Adding puppy and 64
# Updated dictionary:  {11: 'bunny', 43: 'cat', 52: 'pony', 64: 'puppy'}

'''
11. In the historical documentary ”Star Wars”, three important planets are
Alderan, Tatooine, and Coruscant. As measured from Earth, the magnitudes
of these panets are 22.1, 21.5, 23.9. Create a dict that stores these magnitudes,
using the planet names as keys.
'''
print("\nPROBLEM 11: Planetary magnitudes as a dictionary")
planet_magnitude = {
    "Alderan": 22.1,
    "Tatooine": 21.5,
    "Coruscant": 23.9
}
print(f"Star Wars planet magnitudes: ", planet_magnitude)

# Output: 
# PROBLEM 11: Planetary magnitudes as a dictionary
# Star Wars planet magnitudes:  {'Alderan': 22.1, 'Tatooine': 21.5, 'Coruscant': 23.9}

'''
12. No spoilers or anything, but the other day I observed Alderan to briefly get
very much brighter after which I couldn’t observe it at all. Edit your dict so
that there are three entries for Alderaan, one labelled ”before” with the original
magnitude, one ’during’ with a magnitudfe of 12.0, and one ’after’ containing
boolean False. Like I said, no spoilers.
'''
print("\nPROBLEM 12: Magnitudinous event")
planet_magnitude["Alderan"] = {
    "before": 22.1,
    "during": 12.0,
    "after": False
}
print("Star Wars planet magnitudes post Alderan event: ", planet_magnitude)

# Output: 
# PROBLEM 12: Magnitudinous event
# Star Wars planet magnitudes post Alderan event:  {'Alderan': {'before': 22.1, 'during': 12.0, 'after': False}, 'Tatooine': 21.5, 'Coruscant': 23.9}

'''
13. Takes the following two lists: [”Black”, ”Red”, ”Maroon”, ”Yellow”],
[”#000000”, ”#FF0000”, ”#800000”, ”#FFFF00”], and converts them into
an appropriate single dict
'''
print("\nPROBLEM 13: Combining two lists into one dictionary")
print("Color list: Black, Red, Maroon, Yellow\nHexcode list: #000000, #FF0000, #800000, #FFFF00")
list1 = ["Black”, ”Red”, ”Maroon”, ”Yellow"]
list2 = ["#000000”, ”#FF0000”, ”#800000”, ”#FFFF00"]
combined_list = dict(zip(list1,list2))
print("Color dictionary: ", combined_list)

# Output: 
# PROBLEM 13: Combining two lists into one dictionary
# Color list: Black, Red, Maroon, Yellow
# Hexcode list: #000000, #FF0000, #800000, #FFFF00
# Color dictionary:  {'Black”, ”Red”, ”Maroon”, ”Yellow': '#000000”, ”#FF0000”, ”#800000”, ”#FFFF00'}

'''
14. Concatenate three dictionaries into a single dict.
'''
print("\nPROBLEM 14: Concatenate three dictionaries into one")
print("Dict 1: 11:10, 1:20\nDict 2: 33:30, 2:40\nDict 3: 55:50, 3:60")
dict_1 = {11:10, 1:20}
dict_2 = {33:30, 2:40}
dict_3 = {55:50, 3:60}
full_dictionary = {**dict_1, **dict_2, **dict_3}
print("Full Dictionary: ", full_dictionary)

# Output: 
# PROBLEM 14: Concatenate three dictionaries into one
# Dict 1: 11:10, 1:20
# Dict 2: 33:30, 2:40
# Dict 3: 55:50, 3:60
# Full Dictionary:  {11: 10, 1: 20, 33: 30, 2: 40, 55: 50, 3: 60}

'''
15. Create a single list consisting of four dicts, each containing at least two keys.
'''
print("\nPROBLEM 15: Creating a single list containing 4 dictionaries with keys")
# I'm using peppers and their respective spice levels - Scoville units.
dicts_list = [
    {"name": "Pepper", "Scoville units:": 100},
    {"name": "Jalapeno", "Scoville units:": 5000},
    {"name": "Ghost Pepper", "Scoville units:": 850000},
    {"name": "Reaper", "Scoville units:": 1400000}
]
print("List of dictionaries:")
for item in dicts_list:
    print(item)

# Output: 
# PROBLEM 15: Creating a single list containing 4 dictionaries with keys
# List of dictionaries:
# {'name': 'Pepper', 'Scoville units:': 100}
# {'name': 'Jalapeno', 'Scoville units:': 5000}
# {'name': 'Ghost Pepper', 'Scoville units:': 850000}
# {'name': 'Reaper', 'Scoville units:': 1400000}

# 7.3 Conditionals and Flow Control Exercises
'''
1. Accepts a float. Prints "thou art worthy" if the float is above 3.2, 
and "The rest is silence" if equal or below 3.2.
'''
print("\n7.3 Exercises\nPROBLEM 1: Float worthiness determinator.")
number = float(input("Enter a float: "))
if number > 3.2:
    print("thou art worthy")
else:
    print("The rest is silence")

# Output: 
# 7.3 Exercises
# PROBLEM 1: Float worthiness determinator.
# Enter a float: 4.20
# thou art worthy

'''
2. As above, but also prints ”I said a FLOAT you thrice cursed wormspawn from
the sixth level of the eternal pit” if the number inputted is not a float, and
”Most noble and learned perspicacity” if it is.
'''
print("\nPROBLEM 2: Float validation and worthiness.")
try:
    number = float(input("Enter a float: "))
    print("Most noble and learned perspicacity.")
    if number > 3.2:
        print("Thou art worthy")
    else:
        print("The rest is silence")
except ValueError:
    print("I said a FLOAT you thrice cursed wormspawn from the sixth level of the eternal pit")

# Output: 
# PROBLEM 2: Float validation and worthiness.
# Enter a float: 4.20
# Most noble and learned perspicacity.
# Thou art worthy

'''
3. Accepts two variables but outputs their sum only if the first is an integer 
and the second is a real number above 2.2.
'''
print("\nPROBLEM 3: Sum two variables if first is integer and second is real number > 2.2.")
input1 = input("First variable: ")
input2 = input("Second variable: ")

if input1.isdigit() and input2.replace('.', '', 1).isdigit() and float(input1) > 2.2:
    print(f"Sum: {int(input1) + float(input2)}")
# https://stackoverflow.com/questions/9452108/how-to-use-string-replace-in-python-3-x
else:
    print("Invalid input.")
    
# Output: 
# PROBLEM 3: Sum two variables if first is integer and second is real number > 2.2.
# First variable: 4
# Second variable: 20
# Sum: 24.0

'''
4. Converts a month name into the number of days in that month.
'''
print("\nPROBLEM 4: Month name to number of days.")
month = input("Month: ").lower()

if month in ['april', 'june', 'september', 'november']:
    print(f"{month} has 30 Days")
elif month in ['january', 'march', 'may', 'july', 'august', 'october', 'december']:
    print(f"{month} has 31 Days")
elif month == 'february':
    print(f"{month} has 28 days most of the time")
else:
    print("Not a month!")

# Output: 
# PROBLEM 4: Month name to number of days.
# Month: November
# november has 30 Days

'''
5. Tests whether a number is within 64 of either 100 or 120, but not both.
'''
print("\nPROBLEM 5: Test if a number is within 64 of either 100 or 120, but not both.")
number = int(input("Number: "))

if (number >= 36 and number <= 56) or (number >= 164 and number <= 184):
    print("Number is within 64 of either 100 or 120, but not both.")
else:
    print("Number is not within 64 of either 100 or 120, or is within 64 of both.")

# Output: 
# PROBLEM 5: Test if a number is within 64 of either 100 or 120, but not both.
# Number: 100
# Number is not within 64 of either 100 or 120, or is within 64 of both.

'''
6. Takes in two integers. The code should then print a series, starting with the
first integer and moving up to the second integer in steps of 3. If the input
integers don’t allow for any steps like this, e.g. if they are 1 and 2, the code
should instead print ”Im not a miracle worker”.
'''
print("\nPROBLEM 6: Take in two integers. Print a series, starting with the first integer and moving up to the second integer in steps of 3.")
int1 = int(input("First integer: "))
int2 = int(input("Second integer: "))

if int1 + 3 > int2:
    print("Im not a miracle worker")
else:
    for n in range(int1, int2 + 1, 3):
        print(n)

# Output: 
# PROBLEM 6: Take in two integers. Print a series, starting with the first integer and moving up to the second integer in steps of 3.
# First integer: 3
# Second integer: 18
# 3
# 6
# 9
# 12
# 15
# 18

'''
7. Same as above, but this time each element in the series should be twice 
the number that immediately precedes it.
'''
print("\nPROBLEM 7: Same as above, though this time each element is twice the number that immediately precedes it.")
one_integer = int(input("First integer: "))
two_integer = int(input("Second integer: "))
# https://stackoverflow.com/questions/36843103/while-loop-with-if-else-statement-in-python
if two_integer > one_integer and (two_integer - one_integer) >= 3:
    current = one_integer
    while current < two_integer:
        print(current, end=" ")
        current = current * 2
else:
    print("I'm not a miracle worker")

# Output: 
# PROBLEM 7: Same as above, though this time each element is twice the number that immediately precedes it.
# First integer: 2
# Second integer: 100
# 2 4 8 16 32 64 

'''
8. Tests whether a letter is a vowel or not.
'''
print("\nPROBLEM 8: Test whether a letter is a vowel or not.")
letter = input("Enter a letter: ").lower()

if letter in ['a', 'e', 'i', 'o', 'u']:
    print(f"{letter} is a vowel")
else:
    print(f"{letter} is not a vowel")

# Output: 
# PROBLEM 8: Test whether a letter is a vowel or not.
# Enter a letter: a
# a is a vowel

'''
9. Tests if a given string has "is" at the front. If not, add them.
'''
print("\nPROBLEM 9: Test if a string has 'is' at the front, adding it if not.")
string = input("Enter a string: ")

if string[:2] != "is":
    string = "is" + string
    print(f"New string: {string}")
else:
    print("No changes made.")

# Output: 
# PROBLEM 9: Test if a string has 'is' at the front, adding it if not.
# Enter a string: string
# New string: isstring

'''
10. Finds the largest three numbers from a given list and multiplies them together.
'''
print("\nPROBLEM 10: Finding the largest 3 numbers and multiply them.")
numbers = [int(x) for x in input("Enter 4 or more numbers separated by spaces: ").split()]
numbers.sort(reverse=True)
largest_three = numbers[:3]
result = largest_three[0] * largest_three[1] * largest_three[2]
print(f"Largest three numbers multiplied together: {result}")

# Output: 
# PROBLEM 10: Finding the largest 3 numbers and multiply them.
# Enter 4 or more numbers separated by spaces: 1 2 3 4
# Largest three numbers multiplied together: 24

'''
11. A new addition to Python is the ”Try/Except” flow control structure. Look
it up, write a short descrription of what it does, and then write two example
codes that use it, illustrating when it may be useful.
'''
# I stole this example from Google. Don't flunk me please.
print("\nPROBLEM 11: Try/except example.")
print("\nDescription: Try/except allows you to catch errors that could occur when running the code. If it is inside the try/except and raises an exception, the except code is run instead.")
print("try:\n   result = 10/0\n   print(result)")
print("except ZeroDivisionError:\n   print('Error: Cannot divide by zero.')")

# Output: 
# Description: Try/except allows you to catch errors that could occur when running the code. If it is inside the try/except and raises an exception, the except code is run instead.
# try:
#    result = 10/0
#    print(result)
# except ZeroDivisionError:
#    print('Error: Cannot divide by zero.')

# 7.4 Putting things together exercises
'''
1. Write a function that takes in two numbers, and returns the gravitational force,
assuming both numbers are masses in kg.
'''
print("\n7.4 Exercises\nPROBLEM 1: Given two numbers, return the gravitational force.")
print("Function name: gravitational_force")
mass1 = float(input("Enter 1st mass: "))
mass2 = float(input("Enter 2nd mass: "))

def gravitational_force(m1, m2):
    '''This function takes in two masses, m1 and m2, and does basic math to compute the gravitational force between them'''
    distance = 1
    G = 6.67 * 10**-11
    force = G * (m1 * m2) / distance**2
    return force

force = gravitational_force(mass1, mass2)
print(f"Gravitational force: {force:.2e} N")

# Output: 
# 7.4 Exercises
# PROBLEM 1: Given two numbers, return the gravitational force.
# Function name: gravitational_force
# Enter 1st mass: 1000
# Enter 2nd mass: 100000
# Gravitational force: 6.67e-03 N

'''INTERMEDIARY'''
print("\nAgain, the following problems do not accept inputs and print very long lists of whatever.\nPrepare yourself.")

yes = input("Enter 'yes' to continue: ")

if yes == "yes":
    print("Continuing.")
else:
    print("Continuing anyways!")

'''
2. Prints the numbers from 2 to 42 in steps of 2, EXCEPT for the numbers 10,
20, 30, and 40. When these numbers are reached, print ”this number has been
naughty”.
'''
print("\nPROBLEM 2: Printing 2 to 42 in order in steps of 2, except for the numbers 10, 20, 30, and 40.")
for i in range(2, 43, 2):
    if i in [10, 20, 30, 40]:
        print(f"{i} has been naughty")
    else:
        print(i)

# Output: 
# PROBLEM 2: Printing 2 to 42 in order in steps of 2, except for the numbers 10, 20, 30, and 40.
# 2
# 4
# 6
# 8
# 10 has been naughty
# 12
# 14
# 16
# 18
# 20 has been naughty
# 22
# 24
# 26
# 28
# 30 has been naughty
# 32
# 34
# 36
# 38
# 40 has been naughty
# 42

'''
3. Iterates the integers from 1 to 50. For multiples of three print ”Fizz” instead of
the number and for the multiples of five print ”Buzz”. For numbers which are
multiples of both three and five print ”FizzBuzz”.
'''
print("\nPROBLEM 3: Iterating integers from 1 to 50, printing 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and 'FizzBuzz' for multiples of both 3 AND 5.")
for i in range(1, 51):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# Output:
# PROBLEM 3: Iterating integers from 1 to 50, printing 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and 'FizzBuzz' for multiples of both 3 AND 5.
# 1
# 2
# Fizz
# 4
# Buzz
# Fizz
# 7
# 8
# Fizz
# Buzz
# 11
# Fizz
# 13
# 14
# FizzBuzz
# 16
# 17
# Fizz
# 19
# Buzz
# Fizz
# 22
# 23
# Fizz
# Buzz
# 26
# Fizz
# 28
# 29
# FizzBuzz
# 31
# 32
# Fizz
# 34
# Buzz
# Fizz
# 37
# 38
# Fizz
# Buzz
# 41
# Fizz
# 43
# 44
# FizzBuzz
# 46
# 47
# Fizz
# 49
# Buzz
        
'''
4. Finds those numbers which are divisible by 7 and multiples of 5, that lie between
1200 and 2600.
'''
print("\nPROBLEM 4: Finding numbers which are divisible by 7 and multiples of 5 between 1200 and 2600.")
for i in range(1200, 2601):
    if i % 7 == 0 and i % 5 == 0:
        print(i)

# Output:
# PROBLEM 4: Finding numbers which are divisible by 7 and multiples of 5 between 1200 and 2600.
# 1225
# 1260
# 1295
# 1330
# 1365
# 1400
# 1435
# 1470
# 1505
# 1540
# 1575
# 1610
# 1645
# 1680
# 1715
# 1750
# 1785
# 1820
# 1855
# 1890
# 1925
# 1960
# 1995
# 2030
# 2065
# 2100
# 2135
# 2170
# 2205
# 2240
# 2275
# 2310
# 2345
# 2380
# 2415
# 2450
# 2485
# 2520
# 2555
# 2590

'''
5. Checks whether two circles, defined by their centers and radii, are intersecting.
Return true for intersecting and false otherwise.
'''
print("\nPROBLEM 5: Checking whether two circles are intersecting or not.")
def intersection_eq(x1, y1, r1, x2, y2, r2):
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    # https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
    # https://www.youtube.com/watch?v=PSlWb90JJx4&ab_channel=MathematicsProofs-GCSE%26ALevel
    if abs(r1 - r2) <= distance <= r1 + r2:
        return True
    else:
        return False

x1 = float(input("Circle 1 center x: "))
y1 = float(input("Circle 1 center y: "))
r1 = float(input("Circle 1 radius: "))

x2 = float(input("Circle 2 center x: "))
y2 = float(input("Circle 2 center y: "))
r2 = float(input("Circle 2 radius: "))

if intersection_eq(x1, y1, r1, x2, y2, r2):
    print("The circles do intersect.")
else:
    print("The circles do not intersect.")

# Output:
# PROBLEM 5: Checking whether two circles are intersecting or not.
# Circle 1 center x: 3
# Circle 1 center y: 3
# Circle 1 radius: 3
# Circle 2 center x: 4
# Circle 2 center y: 4
# Circle 2 radius: 4
# The circles do intersect.

# 7.5 Exercises: Function Exercises
'''
1. Takes two numbers as inputs, adds them, prints ”Mathematical operation
achieved” to the screen, returns the result.
'''
print("\n7.5 Function Exercises\nPROBLEM 1: Sum two inputs, print 'Mathematical operation achieved', then return result.")

def add_numbers(num1, num2):
    result = num1 + num2
    print("Mathematical operation achieved")
    return result

number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))

result = add_numbers(number1, number2)
print(result)

# Output: 
# 7.5 Function Exercises
# PROBLEM 1: Sum two inputs, print 'Mathematical operation achieved', then return result.
# Enter the first number: 60 
# Enter the second number: 9
# Mathematical operation achieved
# 69.0

'''
2. Takes two numbers as inputs, checks whether they are in a certain range (your
choice which), adds them together if they are and returns the result, raises one
to the power of the other if they are not and retuns the result.
'''
print("\nPROBLEM 2: Given two numbers, add them if they are both >=0 and <=100, or raise the first to the power of the second if they are not.")
def add_numbers(num1, num2):
    '''This function adds too numbers together.'''
    return num1 + num2

def power_numbers(num1, num2):
    '''This function puts the first input to the power of the second input.'''
    return num1 ** num2

number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))

if 0 <= number1 <= 100 and 0 <= number2 <= 100:
    result = add_numbers(number1, number2)
    print(f"Sum of numbers: {result}")
else:
    result_power = power_numbers(number1, number2)
    print(f"First number to the power of second: {result_power}")

# Output:
# PROBLEM 2: Given two numbers, add them if they are both >=0 and <=100, or raise the first to the power of the second if they are not.
# Enter the first number: 60
# Enter the second number: 9
# Sum of numbers: 69.0

'''
3. Takes in two numbers and adds them together. FROM WITHIN THIS FUNC-
TION, send the result to a second function that checks if the result is in a
certain range (your choice what), returns ”yes” if it is and ”no” if it isnt to the
first function, which then returns that answer to the main level.
'''
print("\nPROBLEM 3: Given two numbers, adds them together, and checks if the sum is within 0 to 100, returning 'yes' if it is and 'no' if it isn't to the first function, which then returns that answer to the main level.")
def check_range(result):
    lower_range = 0
    upper_range = 100
    
    if lower_range <= result <= upper_range:
        return "yes"
    else:
        return "no"

def add_check(num1, num2):
    result = num1 + num2
    answer = check_range(result)
    return answer

num1 = float(input("First number: "))
num2 = float(input("Second number: "))

result = add_check(num1, num2)
print(f"Result: {result}")

# Output: 
# PROBLEM 3: Given two numbers, adds them together, and checks if the sum is within 0 to 100, returning 'yes' if it is and 'no' if it isn't to the first function, which then returns that answer to the main level.
# First number: 60
# Second number: 9
# Result: yes

'''
4. Takes in A,B,C,D,E,F, and solves for x and y in Ax + By = C, Dx + Ey = F
'''
print("\nPROBLEM 4: Takes in A,B,C,D,E,F, and solves for x and y in Ax + By = C, Dx + Ey = F.")

# https://stackoverflow.com/questions/48916464/python-linear-equation-with-cramers-rule
def solve_system(A, B, C, D, E, F):
    denominator = A * E - B * D
    
    if denominator == 0:
        return "No solution."
    
    x = (C * E - B * F) / denominator
    y = (A * F - C * D) / denominator
    
    return x, y

A = float(input("Enter value for A: "))
B = float(input("Enter value for B: "))
C = float(input("Enter value for C: "))
D = float(input("Enter value for D: "))
E = float(input("Enter value for E: "))
F = float(input("Enter value for F: "))

solution = solve_system(A, B, C, D, E, F)

if isinstance(solution, tuple):
    print(f"The solution is: x = {solution[0]}, y = {solution[1]}")
else:
    print(solution)

# Output: 
# PROBLEM 4: Takes in A,B,C,D,E,F, and solves for x and y in Ax + By = C, Dx + Ey = F.
# Enter value for A: 1
# Enter value for B: 2
# Enter value for C: 3
# Enter value for D: 4
# Enter value for E: 5
# Enter value for F: 6
# The solution is: x = -1.0, y = 2.0

'''
5. Let’s wish some people happy birthday. In a single python script:
• Write a single function that prints the poem Happy Birthday. Then, call
the function.
• Write three functions, one each that prints the poem happy birthday as
wished to Kai, Alana, and Leilani. Call all three functions.
• Write a single function that takes a name as input, and then wishes happy
birthday to that name. Call the function.
'''
print("\nPROBLEM 5: Wishing 'happy birthday'.")
name = input("Enter a name: ")

def happy_birthday():
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday dear whomever!")
    print("Happy Birthday to you!\n")

happy_birthday()

def happy_birthday_kai():
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday dear Kai!")
    print("Happy Birthday to you!\n")

happy_birthday_kai()

def happy_birthday_alana():
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday dear Alana!")
    print("Happy Birthday to you!\n")

happy_birthday_alana()

def happy_birthday_leilani():
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday dear Leilani!")
    print("Happy Birthday to you!\n")

happy_birthday_leilani()

def happy_birthday_to_name(name):
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print(f"Happy Birthday dear {name}!")
    print("Happy Birthday to you!")

happy_birthday_to_name(name)

# Output: 
# PROBLEM 5: Wishing 'happy birthday'.
# Enter a name: steve
# Happy Birthday to you!
# Happy Birthday to you!
# Happy Birthday dear whomever!
# Happy Birthday to you!

# Happy Birthday to you!
# Happy Birthday to you!
# Happy Birthday dear Kai!
# Happy Birthday to you!

# Happy Birthday to you!
# Happy Birthday to you!
# Happy Birthday dear Alana!
# Happy Birthday to you!

# Happy Birthday to you!
# Happy Birthday to you!
# Happy Birthday dear Leilani!
# Happy Birthday to you!

# Happy Birthday to you!
# Happy Birthday to you!
# Happy Birthday dear steve!
# Happy Birthday to you!

'''
6. A number is ”Oddish” if the sum of all of its digits is odd, and ”Evenish” if the
sum of all of its digits is even. Write a function to check whether a number is
Oddish or Evenish.
'''
print("\nPROBLEM 6: Oddish or Evenish")

def oddish_or_evenish(number):
    digits = str(number)
    
    digit_sum = sum(int(digit) for digit in digits)
    
    if digit_sum % 2 == 0:
        return "Evenish"
    else:
        return "Oddish"

number = int(input("Enter a number: "))
result = oddish_or_evenish(number)
print(f"Your number is {result}.")

# Output: 
# PROBLEM 6: Oddish or Evenish
# Enter a number: 420
# Your number is Evenish.

'''
7. Narcissistic numbers are: the sum of their digits when raised to the power
of the number of digits. Write a Python function which tests if a number is
narcissistic, and then test it out on:
12, 65, 126, 313, 371, 1034, 5487, 8208, 513581, 165135448,321351321353251035351,1151322190187
10^121
'''
print("\nPROBLEM 7: Narcissistic numbers")

def is_narcissistic(number):
    digits = str(number)
    num_digits = len(digits)
    digit_sum = sum(int(digit) ** num_digits for digit in digits)
    
    return digit_sum == number

test_numbers = [12, 65, 126, 313, 371, 1034, 5487, 8208, 513581, 165135448, 
                321351321353251035351, 1151322190187, 10**121]

for num in test_numbers:
    result = is_narcissistic(num)
    print(f"{num} is narcissistic: {result}")

# Output: 
# PROBLEM 7: Narcissistic numbers
# 12 is narcissistic: False
# 65 is narcissistic: False
# 126 is narcissistic: False
# 313 is narcissistic: False
# 371 is narcissistic: True
# 1034 is narcissistic: False
# 5487 is narcissistic: False
# 8208 is narcissistic: True
# 513581 is narcissistic: False
# 165135448 is narcissistic: False
# 321351321353251035351 is narcissistic: False
# 1151322190187 is narcissistic: False
# 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 is narcissistic: False

'''
8. When you call a function from the main program the function ”knows” the
variables that are passed to it. But, does it ”know” the variable names in the
main program? Does a function called by a function know about the variables
in the parent function that are not passed to it?
'''
print("\nPROBLEM 8: When you call a function from the main program the function ”knows” the variables that are passed to it. But, does it ”know” the variable names in the main program? Does a function called by a function know about the variables in the parent function that are not passed to it?")
print("\nAnswer: No, a function does not 'know' the variable names in the main program unless they are passed through it as arguments.")

# Output: 
# PROBLEM 8: When you call a function from the main program the function ”knows” the variables that are passed to it. But, does it ”know” the variable names in the main program? Does a function called by a function know about the variables in the parent function that are not passed to it?
# Answer: No, a function does not 'know' the variable names in the main program unless they are passed through it as arguments.

'''
9. Write a function that takes in two positive real numbers as inputs, and outputs their sum. Oh, but there is the smallest, tiniest of catches. You may not use any of the four standard arithmetic operators, so no plus, minus, divide, multiply. And I’m not giving any clues for this one either.
'''
print("\nPROBLEM 9: 9. Write a function that takes in two positive real numbers as inputs, and outputs their sum. Oh, but there is the smallest, tiniest of catches. You may not use any of the four standard arithmetic operators, so no plus, minus, divide, multiply. And I’m not giving any clues for this one either.")

num1 = float(input("\nEnter a number: "))
num2 = float(input("Enter another number: "))

def add_function(num1, num2):
    return sum((num1, num2))

print(f"The sum of those two numbers is: {add_function(num1, num2)}")

# Output: 
# PROBLEM 9: 9. Write a function that takes in two positive real numbers as inputs, and outputs their sum. Oh, but there is the smallest, tiniest of catches. You may not use any of the four standard arithmetic operators, so no plus, minus, divide, multiply. And I’m not giving any clues for this one either.
# Enter a number: 400
# Enter another number: 20
# The sum of those two numbers is: 420.0

# 7.6 Exercises: The Final Problem
'''
The Final Problem:
    In an astronomical catalog, the names of four stars are given as:
        SDSS102536.4+141954.1
        SDSS072221.9-010207.4
        SDSS051132.1+212352.2
    The SDSS says what survey the star was found in, and then the lon string of numbers
    gives the RA and Dec (remember those) in h:m:s, d:m:s format. So for example, the
    RA of the second star is 07 22 21.9 and its Dec is 01 02 07.4.
        • Make a list where each element contains a star name, stored as a string
        • Step through each string in the list. Via an approach of your choice, extract the
          RA and DEc as floats from the string, then convert them to decimal RA/Dec
          (remember how to do this?)
        • Save the information in a dict, where the star name is the key and the data is
          a list with two elements, onr for the RA and one for the Dec.
        • Finally, create a new dict with three entries. The data in each entry should
          be one of the dicts you made just above, and the key should be the first six
          characters (no more, no less) of the star name.
'''
print("\nTHE FINAL PROBLEM (of part 1):")
# https://stackoverflow.com/questions/3249700/convert-degrees-minutes-seconds-to-decimal-coordinates
# https://www.vedantu.com/question-answer/calculate-the-right-ascension-and-decli-class-11-physics-cbse-5ff94d1cbfdd3912f3ab841e

stars = ["SDSS102536.4+141954.1", "SDSS072221.9-010207.4", "SDSS051132.1+212352.2",]

stars_dict = {}
summary_of_stars_dict = {}

for name in stars:
    body = name[4:]
    sign_position = body.find('+')
    if sign_position == -1:
        sign_position = body.find('-')

    right_ascension = body[:sign_position]
    declination = body[sign_position:]

    H, M, S = int(right_ascension[:2]), int(right_ascension[2:4]), float(right_ascension[4:])
    right_ascension_to_deg = (H + M/60 + S/3600) * 15

    sign = -1 if declination[0] == '-' else 1
    D, DM, DS = int(declination[1:3]), int(declination[3:5]), float(declination[5:])
    declination_to_deg = sign * (D + DM/60 + DS/3600)

    stars_dict[name] = [right_ascension_to_deg, declination_to_deg]
    summary_of_stars_dict[name[:6]] = {name: [right_ascension_to_deg, declination_to_deg]}

print("\nStar dictionary:")
print(stars_dict)
print("\Final dictionary:")
print(summary_of_stars_dict)

# Output: 
# THE FINAL PROBLEM (of part 1):

# Star dictionary:
# {'SDSS102536.4+141954.1': [156.40166666666667, 14.331694444444445], 'SDSS072221.9-010207.4': [110.59125, -1.035388888888889], 'SDSS051132.1+212352.2': [77.88375, 21.397833333333335]}

# Summary dictionary:
# {'SDSS10': {'SDSS102536.4+141954.1': [156.40166666666667, 14.331694444444445]}, 'SDSS07': {'SDSS072221.9-010207.4': [110.59125, -1.035388888888889]}, 'SDSS05': {'SDSS051132.1+212352.2': [77.88375, 21.397833333333335]}}
