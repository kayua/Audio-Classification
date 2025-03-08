# Nome do arquivo onde as linhas serão gravadas
filename = "teste.csv"
LINES = 10000 #ate 5 folds
FEATURES = 10
# Abrir o arquivo para escrita
with open(filename, 'w') as f:
    line = ','.join(["f{}".format(i) for i in range(FEATURES)] ) # 100 zeros separados por vírgulas
    # line = '# '  
    # line += ','.join(['A'] * 100) 
    f.write(line + ',label\n') 
    # Gerar 1000 linhas de 0
    for _ in range(LINES):
        line = ','.join(['0'] * (FEATURES+1))  # 100 zeros separados por vírgulas
        f.write(line + '\n')  # Escrever a linha no arquivo com quebra de linha

        line = ','.join(['1'] * (FEATURES+1))  # 100 uns separados por vírgulas
        f.write(line + '\n')  # Escrever a linha no arquivo com quebra de linha


    # for _ in range(LINES):
    #     #line = ','.join(['0'] * 100)  # 100 zeros separados por vírgulas
    #     #f.write(line + '\n')  # Escrever a linha no arquivo com quebra de linha
    #     line = ','.join(['1'] * 100)  # 100 uns separados por vírgulas
    #     f.write(line + '\n')  # Escrever a linha no arquivo com quebra de linha
        

print(f"Arquivo '{filename}' criado com sucesso!")
