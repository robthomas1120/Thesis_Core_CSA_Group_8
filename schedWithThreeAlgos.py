from datetime import datetime
import time
import random
import copy
import json
import openpyxl
import threading
import json
import random

def initialize_lahc(initial_solution, initial_cost, list_length):
    # Initialize the look-back list with initial_cost repeated list_length times
    lookback_list = [initial_cost] * list_length
    best_solution = initial_solution  # Best solution starts as the initial solution
    best_cost = initial_cost  # Best cost starts as the initial cost
    k = 0  # Counter for iterations
    
    return lookback_list, best_solution, best_cost, k

def initialize_dlas(initial_solution, initial_cost, list_length):
    # Initialize the phi list with initial_cost repeated list_length times
    phi = [initial_cost] * list_length
    best_solution = initial_solution  # Best solution starts as the initial solution
    best_cost = initial_cost  # Best cost starts as the initial cost
    k = 0  # Counter for iterations
    phi_max = initial_cost  # Initialize phi_max as the initial cost
    
    return phi, best_solution, best_cost, k, phi_max

def initialize_schc(initial_solution, initial_cost, max_steps):
    # SCHC Initialization
    step_count = 0
    best_solution = initial_solution
    best_cost = initial_cost
    return best_solution, best_cost, step_count

def perturb(schedule, exams, periods, rooms):
    new_schedule = copy.deepcopy(schedule)
    # Randomly pick two exams to swap periods or rooms
    exam1, exam2 = random.sample(exams, 2)
    # Swap periods or rooms with equal probability
    if random.choice([True, False]):
        new_schedule[exam1['exam_id']], new_schedule[exam2['exam_id']] = new_schedule[exam2['exam_id']], new_schedule[exam1['exam_id']]
    else:
        # Swap rooms between two exams
        period1, room1, students1, day1, duration1, room_penalty1, period_penalty1 = new_schedule[exam1['exam_id']]
        period2, room2, students2, day2, duration2, room_penalty2, period_penalty2 = new_schedule[exam2['exam_id']]
        new_schedule[exam1['exam_id']] = (period1, room2, students1, day1, duration1, room_penalty1, period_penalty1)
        new_schedule[exam2['exam_id']] = (period2, room1, students2, day2, duration2, room_penalty2, period_penalty2)
    
    return new_schedule

def compute_cost(schedule, institutional_weightings, student_counts):
    hard_weight, soft_weight = calculate_schedule_weight(schedule, institutional_weightings, student_counts)
    return hard_weight + soft_weight

def save_schedule_to_excel(schedule, periods, output_file):
    # Create a new workbook and select the active worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Exam Schedule"

    # Add headers to the sheet
    sheet.append(["Exam ID", "Period Date", "Period Time", "Duration (mins)", "Room", "Penalty"])

    # Add exam schedule data
    for exam_id, details in schedule.items():
        period_id = details[0]  # Access period_id from tuple
        room_id = details[1]    # Access room_id from tuple

        # Get period information or mark as "Unscheduled" if not assigned
        if period_id != -1:
            period = periods[period_id]
            period_date = period.get('date', "Unscheduled")
            period_time = period.get('time', "")
            duration = period.get('duration_in_minutes', "")
            penalty = period.get('penalty', "")
        else:
            period_date = "Unscheduled"
            period_time = ""
            duration = ""
            penalty = ""

        # Get room information
        room = room_id if room_id != -1 else "No room"

        # Append data to the sheet
        sheet.append([exam_id, period_date, period_time, duration, room, penalty])

    # Save the workbook to the specified output file
    workbook.save(output_file)
    print(f"Schedule saved to {output_file}")

def LAHC(initial_solution, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    lookback_list, best_solution, best_cost, k = initialize_lahc(initial_solution, initial_cost, list_length)
    current_solution = initial_solution
    current_cost = initial_cost
    
    start_time = time.time()


    # Convergence criterion: count the number of iterations without improvement
    no_improve_count = 0

    while no_improve_count < max_no_improve:
        # Generate a new candidate solution
        new_solution = perturb(current_solution, exams, periods, rooms)
        new_cost = compute_cost(new_solution, institutional_weightings, student_counts)
        
        # Update the best solution if the new solution is better
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
            no_improve_count = 0  # Reset no improvement counter
        else:
            no_improve_count += 1  # Increment no improvement counter

        # Acceptance criterion based on the look-back list
        if new_cost <= lookback_list[k % list_length]:
            current_solution = new_solution
            current_cost = new_cost

        # Update the look-back list with the current cost
        lookback_list[k % list_length] = current_cost
        
        # Increment counter
        k += 1
        
        print(f"Current Best Cost in LAHC: [{best_cost}], No Improvement Count: [{no_improve_count}]")
    
    elapsed_time = time.time() - start_time
    print(f"Convergence took {elapsed_time:.2f} seconds.")
    return best_solution, best_cost

def chatGPTDLAS(initial_solution, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    lookback_list, best_solution, best_cost, k = initialize_lahc(initial_solution, initial_cost, list_length)
    current_solution = initial_solution
    current_cost = initial_cost

    no_improve_count = 0

    start_time = time.time()

    # Iteration: Check convergence
    while no_improve_count < max_no_improve:
        # Perturbation: Generate a new candidate solution
        new_solution = perturb(current_solution, exams, periods, rooms)
        new_cost = compute_cost(new_solution, institutional_weightings, student_counts)

        # Acceptance: Check if the candidate solution meets acceptance criteria
        if new_cost <= lookback_list[k % list_length] or new_cost <= best_cost:
            # Update current schedule and cost
            current_solution = new_solution
            current_cost = new_cost

            # Update best solution if new cost is better
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                no_improve_count = 0  # Reset no improvement counter
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Replacement: Update the lookback list with current cost if needed
        if current_cost > lookback_list[k % list_length]:
            lookback_list[k % list_length] = current_cost

        # Increment the iteration pointer
        k += 1

        print(f"Current Best Cost in DLAS: [{best_cost}], No Improvement Count: [{no_improve_count}]")

    elapsed_time = time.time() - start_time
    print(f"Convergence took {elapsed_time:.2f} seconds.")
    return best_solution, best_cost

# def robDLAS(initial_solution, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
#     lookback_list, best_solution, best_cost, k = initialize_lahc(initial_solution, initial_cost, list_length)
#     current_solution = initial_solution
#     current_cost = initial_cost

#     no_improve_count = 0

#     start_time = time.time()

#     # Iteration: Check convergence
#     while no_improve_count < max_no_improve:
#         # Perturbation: Generate a new candidate solution
#         new_solution = perturb(current_solution, exams, periods, rooms)
#         new_cost = compute_cost(new_solution, institutional_weightings, student_counts)

#         # Acceptance: Check if the candidate solution meets acceptance criteria
#         if new_cost <= lookback_list[k % list_length] or new_cost <= best_cost:
#             # Update current schedule and cost
#             current_solution = new_solution
#             current_cost = new_cost

#             # Update best solution if new cost is better
#             if current_cost < best_cost:
#                 best_solution = current_solution
#                 best_cost = new_cost
#                 no_improve_count = 0  # Reset no improvement counter
#             else:
#                 no_improve_count += 1

#         # Replacement: Update the lookback list with current cost if needed
#         if current_cost > lookback_list[k % list_length]:
#             lookback_list[k % list_length] = current_cost

#         else:
#             if  current_cost < lookback_list[k % list_length]:
#                 if lookback_list[k % list_length] == best_cost:
#                     no_improve_count += 1
#                 else:
#                     lookback_list[k % list_length] = current_cost
#                     no_improve_count = 0
#             # else:
#             #     no_improve_count += 1
        

#         print(f"Current Best Cost in DLAS: [{best_cost}], No Improvement Count: [{no_improve_count}]")

#     elapsed_time = time.time() - start_time
#     print(f"Convergence took {elapsed_time:.2f} seconds.")
#     return best_solution, best_cost



# def DLAS(initial_solution, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
#     lookback_list, best_solution, best_cost, k = initialize_lahc(initial_solution, initial_cost, list_length)
#     current_solution = initial_solution
#     current_cost = initial_cost
    
#     start_time = time.time()


#     # Convergence criterion: count the number of iterations without improvement
#     no_improve_count = 0

#     while no_improve_count < max_no_improve:
#         # Generate a new candidate solution
#         new_solution = perturb(current_solution, exams, periods, rooms)
#         new_cost = compute_cost(new_solution, institutional_weightings, student_counts)
        
#         # Update the best solution if the new solution is better
#         if new_cost < best_cost:
#             best_solution = new_solution
#             best_cost = new_cost
#             no_improve_count = 0  # Reset no improvement counter
#         else:
#             no_improve_count += 1  # Increment no improvement counter

#         # Acceptance criterion based on the look-back list
#         if new_cost < lookback_list[k % list_length]:
#             current_solution = new_solution
#             current_cost = new_cost

#         # Update the look-back list with the current cost
#         lookback_list[k % list_length] = current_cost
        
#         # Increment counter
#         k += 1
        
#         print(f"Current Best Cost in DLAS: [{best_cost}], No Improvement Count: [{no_improve_count}]")
    
#     elapsed_time = time.time() - start_time
#     print(f"Convergence took {elapsed_time:.2f} seconds.")
#     return best_solution, best_cost

def schc(initial_solution, initial_cost, max_steps, max_iterations, convergence_threshold, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    best_solution = initial_solution
    best_cost = initial_cost
    current_solution = initial_solution
    current_cost = initial_cost
    step_count = 0  # Tracks consecutive worse solutions accepted
    iterations_since_last_improvement = 0  # Tracks iterations without improvement

    for iteration in range(max_iterations):
        # Generate a neighboring solution
        new_solution = perturb(current_solution, exams, periods, rooms)
        new_cost = compute_cost(new_solution, institutional_weightings, student_counts)
        
        # If the new solution is better, accept it
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
            current_solution = new_solution
            current_cost = new_cost
            step_count = 0  # Reset step count when an improvement is found
            iterations_since_last_improvement = 0  # Reset convergence counter
        elif step_count < max_steps:
            # Accept worse solution if step count threshold is not reached
            current_solution = new_solution
            current_cost = new_cost
            step_count += 1
            iterations_since_last_improvement += 1
        else:
            # Reset to best solution when step count limit is exceeded
            current_solution = best_solution
            current_cost = best_cost
            step_count = 0  # Reset step count
            iterations_since_last_improvement += 1

        print(f"Current Best Cost in SCHC: [{best_cost}], No Improvement Count: [{iteration + 1}]")
        
        # Check for convergence
        if iterations_since_last_improvement >= convergence_threshold:
            print("Converged after", iteration + 1, "iterations.")
            break
    
    return best_solution, best_cost

def parse_input(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    exams = data['exams']
    periods = data['periods']
    rooms = data['rooms']
    period_constraints = data['period_constraints']
    room_constraints = data['room_constraints']
    institutional_weightings = data['institutional_weightings']

    parsed_periods=[]
    for indexP, periods in enumerate(periods):
         unique_period_id = f"period_{indexP}"
         parsed_periods.append({'period_id': unique_period_id, 
                                 'date': periods["date"],
                                 'time': periods["time"],
                                     'duration_in_minutes': periods["duration_in_minutes"],
                                     'penalty': periods["penalty"]})
    #print(parsed_periods)
    #print(f"PERIOD {indexP}")
    
    # Parse exams and append unique exam_id
    parsed_exams = []
    for indexE, exam in enumerate(exams):
        for exam_id, exam_data in exam.items():
            unique_exam_id = f"exam_{indexE}"
            duration = exam_data[0]
            exam_data.pop(0)  # Removes the first item from the exam_data array
            parsed_exams.append({'exam_id': unique_exam_id, 'duration': duration, 'exam_data': exam_data})
            
        #print(exam_data)

    parsed_rooms=[]
    for indexP, rooms in enumerate(rooms):
         unique_room_id = f"room_{indexP}"
         parsed_rooms.append({'room_id': unique_room_id, 
                                'capacity': rooms["room_capacity"],
                                'penalty': rooms["penalty"]})

    return parsed_exams, parsed_periods, parsed_rooms, period_constraints, room_constraints, institutional_weightings

def check_period_constraints(exam_id, period_id, period_constraints, schedule):
    for constraint in period_constraints:
        if constraint.get('constraint') == 'EXAM_COINCIDENCE':
            if schedule.get(constraint['exam_2'], (-1, -1))[0] == period_id:
                return False
        elif constraint.get('constraint') == 'EXCLUSION':
            if schedule.get(constraint['exam_2'], (-1, -1))[0] == period_id:
                return False
        elif constraint.get('constraint') == 'AFTER':
            scheduled_period = schedule.get(constraint['exam_2'], (-1, -1))[0]
            if scheduled_period != -1 and scheduled_period >= period_id:
                return False
    return True

def check_room_constraints(exam_id, room_id, room_constraints, schedule):
    for constraint in room_constraints:
        if constraint.get('constraint') == 'ROOM_EXCLUSIVE' and constraint['exam'] == exam_id:
            if room_id in [room for _, room in schedule.values()]:
                return False
    return True

def generate_schedule(exams, periods, rooms, period_constraints, room_constraints, schedule, index=0):
    if index == len(exams):
        return True  # All exams scheduled successfull

    exam_id = exams[index]['exam_id']
    exam_data = exams[index]['exam_data']
    exam_duration = exams[index]['duration']

    period_ids = list(range(len(periods)))
    room_ids = list(range(len(rooms)))
    
    random.shuffle(period_ids)  # Shuffle to generate schedules
    random.shuffle(room_ids)


    #print(f"Trying to schedule exam {exam_id}")

    for period_id in period_ids:
        if not check_period_constraints(exam_id, period_id, period_constraints, schedule):
            continue
        
        for room_id in room_ids:
            if not check_room_constraints(exam_id, room_id, room_constraints, schedule):
                continue
            
            # Check if the period and room are already occupied
            if any(sched == (period_id, room_id) for sched in schedule.values()):
                continue

            #print(f"PERIOD PENALTY  {rooms[room_id]['penalty']}")
            schedule[exam_id] = (period_id, room_id, exam_data, periods[period_id]['date'], exam_duration, rooms[room_id]['penalty'], periods[period_id]['penalty'])
           # print(f"Scheduled exam {exam_id} to period {period_id} and room {room_id}")
            
            if generate_schedule(exams, periods, rooms, period_constraints, room_constraints, schedule, index + 1):
                return True
            
            # If scheduling fails, backtrack
            # print(f"Backtracking on exam {exam_id}")
            schedule[exam_id] = (-1, -1, exam_data[0])  # Backtrack
    
    return False  # If no valid schedule found

def calculate_schedule_weight(schedule, institutional_weightings, student_counts):
    hard_constraints_penalty = 0
    soft_constraints_penalty = 0
    periods_diff = 0
    common_students = 0
    #sorted_schedule = sorted(schedule.items(), key=lambda x: x[1][0], reverse=False)
    #print(sorted_schedule)
    
    # Hard Constraints Violations
    # Check conflicts: Two conflicting exams in the same period
    for exam1, (period1, room1, students, day1, duration, room_penalty, period_penalty) in schedule.items():
        for exam2, (period2, room2, students,day2, duration, room_penalty, period_penalty) in schedule.items():
            if exam1 != exam2 and period1 == period2:
                hard_constraints_penalty += 1  # Increment for each conflict

    # Soft Constraints Violations
    for weighting in institutional_weightings:
        if weighting['type'] == 'TWOINAROW':
            for exam1, (period1, _, students1, day1, _, _, _) in schedule.items():
                for exam2, (period2, _, students2, day2, _, _, _) in schedule.items():
                    # Only check each pair once by ensuring exam2 is checked after exam1
                    if exam1 != exam2 and period2 > period1 and abs(period1 - period2) == 1 and day1 == day2:  # Check if back-to-back
                        common_students = set(students1).intersection(set(students2))
                        # if len(common_students) != 0:
                        #     print(f"[{exam1}] and [{exam2}] are happening simultaneously")
                        #     print(f" there are [{len(common_students)}] students taking them, multiply with {weighting['numbers'][0]}, and add to [{soft_constraints_penalty}]")
                        soft_constraints_penalty += len(common_students) * weighting['numbers'][0]         
            #print(f"Soft Constraint Total after [TWOINAROW] is: {soft_constraints_penalty}")
        
        elif weighting['type'] == 'TWOINADAY':
            for exam1, (period1, _,students1,day1, _, _, _) in schedule.items():
                for exam2, (period2, _,students2,day2, _, _, _) in schedule.items():
                    if exam1 != exam2 and day1 == day2 and period2>period1 and abs(period1 - period2) > 1:
                        common_students = set(students1).intersection(set(students2))
                        # if len(common_students) != 0:
                        #      print(f"[{exam1}] and [{exam2}] are happening in the same day [{day1}] = [{day2}]")
                        #      print(f" there are [{len(common_students)}] students taking them, multiply with {weighting['numbers'][0]}, and add to [{soft_constraints_penalty}]")
                        soft_constraints_penalty += len(common_students) * weighting['numbers'][0]
            #print(f"Soft Constraint Total after [TWOINADAY] is: {soft_constraints_penalty}")

        elif weighting['type'] == 'PERIODSPREAD':
            period_spread = weighting['numbers'][0]
            for exam1, (period1, _,students1,day1, _, _, _) in schedule.items():
                for exam2, (period2, _,students2,day1, _, _, _) in schedule.items():
                    if day2 > day1 or (day2 == day1 and period2 > period1):
                        periods_diff = ((datetime.strptime(day2, "%d:%m:%Y") - datetime.strptime(day1, "%d:%m:%Y"))).days * 3 + (period2 - period1)
                        if abs(periods_diff) <= period_spread:
                            common_students = set(students1).intersection(students2)
                            #if len(common_students) != 0:
                                # print(f"[{exam1}] and [{exam2}] are happening within the period spread, starting from [{day1}], ending on [{day2}]")
                                # print(f"[{exam1}] is during period [{period1}], and [{exam2}] is during period [{period2}]")
                                # print(f"There are {len(common_students)} that have multiple exams within the period spread, add to [{soft_constraints_penalty}] ")
                                # print(f" [{exam1}] : [{students1}]")
                                # print(f" [{exam2}] : [{students2}]")
                                # print(f" [common] : [{common_students}]")
                            soft_constraints_penalty += len(common_students)
            #print(f"Soft Constraint Total after [PERIODSPREAD] is: {soft_constraints_penalty}")

        elif weighting['type'] == 'NONMIXEDDURATIONS':
            duration_count = {}
            for exam, (period, room,_,_,duration, _, _) in schedule.items():
                if (period, room) not in duration_count:
                    duration_count[(period, room)] = set()
                duration_count[(period, room)].add(duration)
            for (period, room), durations in duration_count.items():
                if len(durations) > 1:
                    soft_constraints_penalty += (len(durations) - 1) * weighting['numbers'][0]

           # print(f"Soft Constraint Total after [NONMIXEDDURATIONS] is: {soft_constraints_penalty}")


        elif weighting['type'] == 'FRONTLOAD':
             # Initialize a set to store unique days
            unique_days = set()

            # Loop through the schedule to collect unique days
            for exam, (period, room, students, day, duration, _, _) in schedule.items():
                unique_days.add(day)  # Add the day to the set

            # Calculate the number of unique days
            number_of_days_in_schedule = len(unique_days)

            num_largest = weighting['numbers'][0]  # Number of largest exams to consider
            num_last_periods = weighting['numbers'][1]  # Number of last periods to check

            exams = [(exam, (period, room, students, day, duration, _, _)) for exam, (period, room, students, day, duration, _, _) in schedule.items()]

            # Sort exams by the number of students, descending
            sorted_exams = sorted(exams, key=lambda x: len(x[1][2]), reverse=True)  # x[1][2] accesses the students list
            largest_exams = sorted_exams[:num_largest]  # Get the largest exams

            # Define the last date in the schedule
            last_period_date = max(datetime.strptime(day, "%d:%m:%Y") for _, (_, _, _, day, _, _, _) in schedule.items())
            #print(f"Last Period: {last_period_date}")
            
            # Debug: print the largest exams being considered
            # print("Largest exams considered for FRONTLOAD penalty:")
            # for exam, (period, room, students, day, duration, _, _) in largest_exams:
            #     print(f"Exam: {exam}, Period: {period}, Room: {room}, Day: {day}, Duration: {duration}, Students: {len(students)}")

            # Calculate penalty for each of the largest exams
            for exam, (period, room, students, day, duration, _, _) in largest_exams:
                exam_date = datetime.strptime(day, "%d:%m:%Y")  # Convert exam day to datetime
                days_from_last_period = (last_period_date - exam_date).days  # Calculate days difference
                
                #Debug: print days from last period
                #print(f"Exam Date: {exam_date}, Last Period Date: {last_period_date}, Days from Last Period: {days_from_last_period}, Current Period: {period}")

                # Determine if the exam is within the last few periods
                if days_from_last_period * 3 + period >= (number_of_days_in_schedule * 3 - num_last_periods):
                    penalty_value = weighting['numbers'][2]
                    soft_constraints_penalty += penalty_value
                    
                    #Print what is being added to the penalty
                    #print(f"Added penalty for {exam}: {penalty_value} (Total now: {soft_constraints_penalty})")

            #print(f"Soft Constraint Total after [FRONTLOAD] is: {soft_constraints_penalty}")


    #Room Penalties
    for exam, (period, room, _, _, duration, room_penalty,_) in schedule.items():
        soft_constraints_penalty += room_penalty

    #print(f"Soft Constraint Total after [ROOMPENALTY] is: {soft_constraints_penalty}")
            
    #Period Penalties     
    for exam, (period, room, _, _, duration, _, period_penalty) in schedule.items():
        soft_constraints_penalty += period_penalty

    #print(f"Soft Constraint Total after [PERIODPENALTY] is: {soft_constraints_penalty}")

    return hard_constraints_penalty, soft_constraints_penalty

def generate_n_schedules(n, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    schedules = []
    attempts = 0

    while len(schedules) < n and attempts < n * 10:  # Limit to avoid infinite loops
        attempts += 1
        schedule = {exam['exam_id']: (-1, -1) for exam in exams}
        
        print(f"Attempting to generate schedule {attempts}")
        
        if generate_schedule(exams, periods, rooms, period_constraints, room_constraints, schedule):
            if schedule not in schedules:
                schedules.append(copy.deepcopy(schedule))
        else:
            print("Could not generate a valid schedule in this attempt.")

    return schedules
def run_lahc(initial_schedule, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    print("Running LAHC to optimize the schedule...")
    best_schedule, best_cost = LAHC(
        initial_schedule, initial_cost, list_length, max_no_improve,
        compute_cost, exams, periods, rooms, period_constraints, room_constraints,
        institutional_weightings, student_counts
    )
    with open("optimized_LAHC_output.txt", "w") as file:
        file.write(f"Best schedule cost: {best_cost}\n")
        file.write(json.dumps(best_schedule, indent=4))
    save_schedule_to_excel(best_schedule, periods, "optimized_LAHC_schedule.xlsx")
    print(f"LAHC best schedule cost: {best_cost}")

def run_dlas(initial_schedule, initial_cost, list_length, max_no_improve, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    print("Running DLAS to optimize the schedule...")
    best_schedule, best_cost = chatGPTDLAS(
        initial_schedule, initial_cost, list_length, max_no_improve,
        compute_cost, exams, periods, rooms, period_constraints, room_constraints,
        institutional_weightings, student_counts
    )
    with open("optimized_DLAS_output.txt", "w") as file:
        file.write(f"Best schedule cost: {best_cost}\n")
        file.write(json.dumps(best_schedule, indent=4))
    save_schedule_to_excel(best_schedule, periods, "optimized_DLAS_schedule.xlsx")
    print(f"DLAS best schedule cost: {best_cost}")

def run_schc(initial_schedule, initial_cost, max_steps, max_iterations, convergence_threshold, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    print("Running SCHC to optimize the schedule...")
    best_schedule, best_cost = schc(
        initial_schedule, initial_cost, max_steps, max_iterations, convergence_threshold,
        compute_cost, exams, periods, rooms, period_constraints, room_constraints,
        institutional_weightings, student_counts
    )
    with open("optimized_SCHC_output.txt", "w") as file:
        file.write(f"Best schedule cost: {best_cost}\n")
        file.write(json.dumps(best_schedule, indent=4))
    save_schedule_to_excel(best_schedule, periods, "optimized_SCHC_schedule.xlsx")
    print(f"SCHC best schedule cost: {best_cost}")

def main():
    json_file = "/Users/robalvarez/Desktop/Thesis_Core_CSA_Group_8/examtojson1.json"  # Your file path here
    exams, periods, rooms, period_constraints, room_constraints, institutional_weightings = parse_input(json_file)
    
    # Placeholder for student counts
    student_counts = {exam['exam_id']: random.randint(1, 30) for exam in exams}
    
    # Initial schedule generation (try to generate one valid schedule first)
    initial_schedule = {exam['exam_id']: (-1, -1) for exam in exams}
    if not generate_schedule(exams, periods, rooms, period_constraints, room_constraints, initial_schedule):
        print("Failed to generate an initial schedule.")
        return
    
    # Initial cost
    initial_cost = compute_cost(initial_schedule, institutional_weightings, student_counts)
    
    # Save the original schedule before optimization
    with open("original_schedule_output.txt", "w") as file:
        file.write("Original schedule before optimization:\n")
        file.write(json.dumps(initial_schedule, indent=4))
        file.write(f"\nOriginal schedule cost: {initial_cost}\n")

    # Parameters for each algorithm
    lahc_params = (initial_schedule, initial_cost, 10, 100, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts)
    dlas_params = (initial_schedule, initial_cost, 5, 50, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts)
    schc_params = (initial_schedule, initial_cost, 5, 500, 100, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts)
    
    # Create threads for each algorithm
    lahc_thread = threading.Thread(target=run_lahc, args=lahc_params)
    dlas_thread = threading.Thread(target=run_dlas, args=dlas_params)
    schc_thread = threading.Thread(target=run_schc, args=schc_params)
    
    # Start the threads
    lahc_thread.start()
    dlas_thread.start()
    schc_thread.start()
    
    # Wait for all threads to complete
    lahc_thread.join()
    dlas_thread.join()
    schc_thread.join()
    
    print("All algorithms have completed execution.")

if __name__ == "__main__":
    main()