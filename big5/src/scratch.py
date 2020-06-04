fin = open("text_tokens.csv", "r")

fout = open("ttt.csv", "w")

header = fin.readline()
line = header.rstrip("\n")

fout.write(line+",Tweet_id,User_id\n")
counter = 0
for ln in fin.readlines():
    new = ln.rstrip("\n")
    fout.write(new+","+str(counter)+","+str(counter+15000)+"\n")
    counter = counter+1
fout.close()
fin.close()