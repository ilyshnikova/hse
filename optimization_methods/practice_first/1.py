def check_state(field, i, j):
	num_stable = 0
	num_unstable = 0
	if (j - 1 >= 0):
		if (field[i][j - 1] == 2):
			num_stable += 1
		if (field[i][j - 1] == 3):
			num_unstable += 1
	if (i - 1 >= 0):
		if (field[i - 1][j] == 2):
			num_stable += 1
		if (field[i - 1][j] == 3):
			num_unstable += 1
	if (j + 1 < len(field[0])):
		if (field[i][j + 1] == 2):
			num_stable += 1
		if (field[i][j + 1] == 3):
			num_unstable += 1
	if (i + 1 < len(field)):
		if (field[i + 1][j] == 2):
			num_stable += 1
		if (field[i + 1][j] == 3):
			num_unstable += 1

	if (num_stable > 1):
		return 2

	if (num_unstable + num_stable > 0):
		return 3
	return 1


n, m, k = list(map(int, input().split()))
field = []
for i in range(n):
	row = list(map(int, input().split()))
	field.append(row)
changes = [[0 for j in range(m)] for i in range(n)]
cur_field = [[0 for j in range(m)] for i in range(n)]

for step in range (k):
	for i in range(n):
		for j in range(m):
			cur_state = check_state(field, i, j)
			if (cur_state != field[i][j]):
				changes[i][j] += 1
			cur_field[i][j] = cur_state
	field = [[cur_field[i][j] for j in range(m)] for i in range(n)]
#print (changes)
for v in changes:
    print(*v)

