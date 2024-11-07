import json
import random
import time
import copy
import openpyxl

def initialize_dlas(initial_solution, initial_cost, list_length):
    # Initialize the phi list with initial_cost repeated list_length times
    phi = [initial_cost] * list_length
    best_solution = initial_solution  # Best solution starts as the initial solution
    best_cost = initial_cost  # Best cost starts as the initial cost
    k = 0  # Counter for iterations
    phi_max = initial_cost  # Initialize phi_max as the initial cost
    
    return phi, best_solution, best_cost, k, phi_max

def perturb(schedule, exams, periods, rooms):
    new_schedule = copy.deepcopy(schedule)
    # Randomly pick two exams to swap periods or rooms
    exam1, exam2 = random.sample(exams, 2)
    # Swap periods or rooms with equal probability
    if random.choice([True, False]):
        new_schedule[exam1['exam_id']], new_schedule[exam2['exam_id']] = new_schedule[exam2['exam_id']], new_schedule[exam1['exam_id']]
    else:
        # Swap rooms between two exams
        period1, room1 = new_schedule[exam1['exam_id']]
        period2, room2 = new_schedule[exam2['exam_id']]
        new_schedule[exam1['exam_id']] = (period1, room2)
        new_schedule[exam2['exam_id']] = (period2, room1)
    
    return new_schedule

def compute_cost(schedule, institutional_weightings, student_counts):
    hard_weight, soft_weight = calculate_schedule_weight(schedule, institutional_weightings, student_counts)
    return hard_weight + soft_weight

def DLAS(initial_solution, initial_cost, list_length, max_time, compute_cost, exams, periods, rooms, period_constraints, room_constraints, institutional_weightings, student_counts):
    start_time = time.time()
    phi, best_solution, best_cost, k, phi_max = initialize_dlas(initial_solution, initial_cost, list_length)
    current_solution = initial_solution
    current_cost = initial_cost
    
    while time.time() - start_time < max_time:
        new_solution = perturb(current_solution, exams, periods, rooms)
        new_cost = compute_cost(new_solution, institutional_weightings, student_counts)
        
        # Acceptance logic
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
        
        if new_cost < phi[k % list_length]:
            current_solution = new_solution
            current_cost = new_cost
        
        # Handle case where there are no valid indices to choose from
        valid_indices = [i for i in range(len(phi)) if phi[i] != best_cost]
        if valid_indices:
            random_index = random.choice(valid_indices)
            phi[random_index] = new_cost
        else:
            print("No valid indices in phi to choose from.")
        
        if new_cost == phi_max:
            phi_max = max(phi)
        
        k += 1
    
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
    
    # Parse exams and append unique exam_id
    parsed_exams = []
    for index, exam in enumerate(exams):
        for exam_id, exam_data in exam.items():
            unique_exam_id = f"exam_{index}"
            parsed_exams.append({'exam_id': unique_exam_id, 'exam_data': exam_data})
    
    return parsed_exams, periods, rooms, period_constraints, room_constraints, institutional_weightings

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
        return True  # All exams scheduled successfully

    exam_id = exams[index]['exam_id']
    
    period_ids = list(range(len(periods)))
    room_ids = list(range(len(rooms)))
    
    random.shuffle(period_ids)  # Shuffle to generate different schedules
    random.shuffle(room_ids)

    print(f"Trying to schedule exam {exam_id}")

    for period_id in period_ids:
        if not check_period_constraints(exam_id, period_id, period_constraints, schedule):
            continue
        
        for room_id in room_ids:
            if not check_room_constraints(exam_id, room_id, room_constraints, schedule):
                continue
            
            # Check if the period and room are already occupied
            if any(sched == (period_id, room_id) for sched in schedule.values()):
                continue
            
            schedule[exam_id] = (period_id, room_id)
            print(f"Scheduled exam {exam_id} to period {period_id} and room {room_id}")
            
            if generate_schedule(exams, periods, rooms, period_constraints, room_constraints, schedule, index + 1):
                return True
            
            # If scheduling fails, backtrack
            print(f"Backtracking on exam {exam_id}")
            schedule[exam_id] = (-1, -1)  # Backtrack
    
    return False  # If no valid schedule found

def calculate_schedule_weight(schedule, institutional_weightings, student_counts):
    hard_constraints_penalty = 0
    soft_constraints_penalty = 0
    
    # Hard Constraints Violations
    # Check conflicts: Two conflicting exams in the same period
    for exam1, (period1, room1) in schedule.items():
        for exam2, (period2, room2) in schedule.items():
            if exam1 != exam2 and period1 == period2:
                hard_constraints_penalty += 1  # Increment for each conflict

    # Soft Constraints Violations
    for weighting in institutional_weightings:
        if weighting['type'] == 'TWOINAROW':
            for exam1, (period1, _) in schedule.items():
                for exam2, (period2, _) in schedule.items():
                    if exam1 != exam2 and abs(period1 - period2) == 1:  # Check if back-to-back
                        num_students = student_counts.get(exam1, 0)  # Adjust to get correct student count
                        soft_constraints_penalty += num_students * weighting['numbers'][0]

        elif weighting['type'] == 'TWOINADAY':
            for exam1, (period1, _) in schedule.items():
                for exam2, (period2, _) in schedule.items():
                    if exam1 != exam2 and period1 // 3 == period2 // 3 and abs(period1 - period2) > 1:  # Same day, not back-to-back
                        num_students = student_counts.get(exam1, 0)
                        soft_constraints_penalty += num_students * weighting['numbers'][0]

        elif weighting['type'] == 'PERIODSPREAD':
            period_spread = weighting['numbers'][0]
            for exam1, (period1, _) in schedule.items():
                for exam2, (period2, _) in schedule.items():
                    if exam1 != exam2 and abs(period1 - period2) <= period_spread:
                        num_students = student_counts.get(exam1, 0)
                        soft_constraints_penalty += num_students

        elif weighting['type'] == 'MIXEDDURATIONS':
            duration_count = {}
            for exam, (period, room) in schedule.items():
                if room not in duration_count:
                    duration_count[room] = set()
                duration_count[room].add(exam['exam_data']['duration'])  # Assuming duration is in exam_data
            for room, durations in duration_count.items():
                if len(durations) > 1:
                    soft_constraints_penalty += (len(durations) - 1) * weighting['numbers'][0]

        elif weighting['type'] == 'LARGEREXAMS':
            # Assuming 'class_size' indicates the number of students in the exam
            for exam, (period, _) in schedule.items():
                if exam['exam_data']['class_size'] > weighting['numbers'][0] and period >= len(schedule) - weighting['numbers'][1]:
                    soft_constraints_penalty += weighting['numbers'][2]

        elif weighting['type'] == 'ROOMPENALTY':
            for room_id in set(room for _, (_, room) in schedule.items()):
                penalty_value = weighting['numbers'][0]  # Assuming penalty is the first number
                usage_count = sum(1 for (_, room) in schedule.items() if room == room_id)
                soft_constraints_penalty += penalty_value * usage_count

        elif weighting['type'] == 'PERIODPENALTY':
            for period_id in set(period for period, _ in schedule.values()):
                penalty_value = weighting['numbers'][0]  # Assuming penalty is the first number
                usage_count = sum(1 for period, _ in schedule.values() if period == period_id)
                soft_constraints_penalty += penalty_value * usage_count

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

def save_schedule_to_excel(schedule, periods, output_file):
    # Create a new workbook and select the active worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Exam Schedule"

    # Add headers to the sheet
    sheet.append(["Exam ID", "Period Date", "Period Time", "Duration (mins)", "Room", "Penalty"])

    # Add exam schedule data
    for exam_id, (period_id, room_id) in schedule.items():
        if period_id != -1:
            period = periods[period_id]
            period_date = period['date']
            period_time = period['time']
            duration = period['duration_in_minutes']
            penalty = period['penalty']
        else:
            period_date = "Unscheduled"
            period_time = ""
            duration = ""
            penalty = ""

        room = room_id if room_id != -1 else "No room"

        # Append the formatted data
        sheet.append([exam_id, period_date, period_time, duration, room, penalty])

    # Save the workbook to the specified output file
    workbook.save(output_file)
    print(f"Schedule saved to {output_file}")


def main():
    json_file = "/Users/robalvarez/Desktop/thesis/examtojson2.json"  # Your file path here
    exams, periods, rooms, period_constraints, room_constraints, institutional_weightings = parse_input(json_file)
    
    # Placeholder for student counts
    student_counts = {exam['exam_id']: random.randint(1, 30) for exam in exams}
    
    print(f"Loaded {len(exams)} exams, {len(periods)} periods, and {len(rooms)} rooms.")
    
    # Initial schedule generation (try to generate one valid schedule first)
    initial_schedule = {exam['exam_id']: (-1, -1) for exam in exams}
    if not generate_schedule(exams, periods, rooms, period_constraints, room_constraints, initial_schedule):
        print("Failed to generate an initial schedule.")
        return
    
    # Initial cost
    initial_cost = compute_cost(initial_schedule, institutional_weightings, student_counts)
    
    # Save the original schedule before optimization
    with open("original_schedule_output.txt", "w") as file:
        file.write("Original schedule before DLAS optimization:\n")
        file.write(json.dumps(initial_schedule, indent=4))
        file.write(f"\nOriginal schedule cost: {initial_cost}\n")  # Save the original cost

    # Save the original schedule to Excel
    save_schedule_to_excel(initial_schedule, periods, "original_schedule.xlsx")
    
    # DLAS parameters
    list_length = 100
    max_time = 30  # Run for 1 minute
    
    print("Running DLAS to optimize the schedule...")
    
    # Run DLAS for 1 minute
    best_schedule, best_cost = DLAS(
        initial_schedule, initial_cost, list_length, max_time, 
        compute_cost, exams, periods, rooms, period_constraints, room_constraints, 
        institutional_weightings, student_counts
    )
    
    # Output best schedule
    print(f"Best schedule found with cost: {best_cost}")
    print("Best Schedule:", best_schedule)
    
    # Optionally save result to a file
    with open("optimized_schedule_output.txt", "w") as file:
        file.write(f"Best schedule cost: {best_cost}\n")
        file.write(json.dumps(best_schedule, indent=4))
    
    # Save the optimized schedule to Excel
    save_schedule_to_excel(best_schedule, periods, "optimized_schedule.xlsx")

if __name__ == "__main__":
    main()