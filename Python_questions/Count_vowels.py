word = "won a hackathon this week"
original = word
count = 0

for ch in word:
      if ch == "a" or ch == "e" or ch == "i" or ch == "o" or ch == "u":
        # cleaner version - if ch in "aeiou":
        count += 1 

print(f"Total vowels in {original} are {count}")