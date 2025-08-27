'''
1. Write a Python script which computes the gravitational force between (a) the
Earth and the Moon, and (b) The Sun and the Moon (you will need to look
up masses and mean distances). The code should print the result as a number
formatted to two decimal places. This last step involves looking up how to
output formatted numbers in Python.
'''

mass_earth = 5.972 * (10**24)
mass_moon = 7.34767309 * (10**22)
mass_sun = 1.989 * (10**30)
distance_earth_moon = 3.844 * (10**5)
distance_sun_moon = 1.5 * (10**8)
gravitational_constant = 6.6743 * (10**-11)
gravity_earth_moon = (gravitational_constant * mass_earth * mass_moon) / (distance_earth_moon ** 2)
gravity_sun_moon = (gravitational_constant * mass_sun * mass_moon) / (distance_sun_moon ** 2)
# print(f"{distance_earth_moon:.2e}")
# print(f"{distance_sun_moon:.2e}")
print("\nPROBLEM 1:")
print(f"The mass of the earth is {mass_earth:.2e} kg.")
print(f"The mass of the moon is {mass_moon:.2e} kg.")
print(f"The distance between the Earth and the Moon is {distance_earth_moon:.2e} km.")
print(f"The distance between the Sun and the Moon is {distance_sun_moon:.2e} km.")
print(f"The gravitational constant is {gravitational_constant:.2e} m^3 kg^-1 s^-2")
print(f"The gravitational force between the Earth and the Moon is {gravity_earth_moon:.2e}.")
print(f"The gravitational force between the Sun and the Moon is {gravity_sun_moon:.2e}.")

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
electrostatic_force = ((coulomb_constant) * (q1_q2_charge_value)) / (distance_q1_q2 ** 2)
print(f"\nPROBLEM 2:")
print(f"The charge of the first point is {q1_charge:.2e} C.")
print(f"The charge of the second point is {q2_charge:.2e} C.")
print(f"The distance between the first and second points is {distance_q1_q2:.3} m.")
print(f"The electrostatic force between the two points is {electrostatic_force:.3} N.")