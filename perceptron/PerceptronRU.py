import random
from src.UI.InterfaceUsuarioBase import InterfaceUsuarioBase


class PerceptronRU:
    """
    Classe do perceptron que contem todos as regras de negocio que são os algoritmos de treinamento e ajustes
    """
    def __init__(self,interface_usuario: InterfaceUsuarioBase,ru_aluno_referencia:str, num_entradas=7, taxa_aprendizado=0.1):
        """
        Inicializa o Perceptron para identificação de RU
        
        Args:
            num_entradas: Número de entradas (7 dígitos do RU)
            taxa_aprendizado: Taxa de aprendizado (η)
        """
        self.num_entradas = num_entradas
        self.taxa_aprendizado = taxa_aprendizado
        self.treinamento_sucesso = False
  
        # Inicialização aleatória dos pesos entre -0.1 e 0.1
        self.pesos = [random.uniform(-0.1, 0.1) for _ in range(num_entradas)]
        self.bias = random.uniform(-2.0, -1.0)  # Inicializa bias entre -2.0 e -1.0
    
        # RU de referência
        self.ru_referencia = list(map(int, ru_aluno_referencia))
      
        # Inicializa a UI de interação com o User. Obs : Só está implementado a console UI
        self.interface_usuario = interface_usuario
        self.interface_usuario.escrever("\n")
        self.interface_usuario.escrever("Perceptron Iniciado")
        self.interface_usuario.escrever(f"RU de Referência: {self.ru_referencia}")
        self.interface_usuario.escrever(f"Pesos iniciais: {[round(w, 3) for w in self.pesos]}")
        self.interface_usuario.escrever(f"Bias inicial: {self.bias}")
        self.interface_usuario.escrever(f"Taxa de aprendizado: {self.taxa_aprendizado}")
        self.interface_usuario.escrever("\n")
 

    def definir_saida_desejada(self, entradas: list[float]):

        # Transforma a lista de entradas
        # e o RU do aluno em uma RU que representa o valor float
        # e compara para verificar se 

        ru_entrada_neuronio = float("".join(map(str, entradas)))
        ru_aluno = float("".join(map(str, self.ru_referencia)))

        return 1 if ru_entrada_neuronio >= ru_aluno else -1

    
    def calcular_net(self, entradas: list[float]):
        """
        Calcula a entrada (net - Média ponderada das entradas em relacao aos 
        seus respectivos pesos mais o valor de ajuste do bias)
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Valor da net calculado para os pesos e bias atual para epoca
        """

   
        #O método zip irá utilizar um iterador que retornar uma tupla
        #contendo o valor atual do apontamento na lista de entradas e pesos
        #dessa forma a cada iteração terei como retorno a entrada e seu respectivo
        #peso
        net = sum(entrada * peso for entrada, peso in zip(entradas, self.pesos)) + self.bias

        return net

    def funcao_ativacao(self, net):
        """
        Função de ativação degrau unitário
        
        Args:
            net: Valor da entrada
            
        Returns:
            1 se a função de ativação está ativa ou -1 se não está
        """
        return 1 if net >= 0 else -1

    def predizer(self, entradas: list[float]):
        """
        Faz uma predição para as entradas com o intuito de obter se o neuronio está
        ativo ou não.
        Importante : Deve ser usado após o treinamento do neuronio no fluxo de teste
        para validar se o valor retornado por essa predição é igual a saida desejada
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Saída do neurônio (+1 ou -1)
        """
        net = self.calcular_net(entradas)
        return self.funcao_ativacao(net)
    
    def treinar_epoca(self, conjunto_treinamento: list[list[float]]):
        """
        Treina o perceptron por uma época ou seja passando por todo
        conjunto de dados de treinamento
        
        Args:
            conjunto_treinamento: lista de RU's com os padrões de dados
            contendo a diversificação de dados que indica dados validos
            ou não (Ru superiores e inferiores)
            Importante : Cada RU é composto por uma lista de 7 numeros que
            corresponde as entradas do neuronio
            
        Returns:
            Número de erros na época
        """
        erro_epoca = 0
        
        # Enumerate e uma função para retornar o dado e seu indice em um array.
        # Obtendo o indice somente oara exibir na interface do usuario qual padrão RU foi treinado na epoca
        for i, entradas in enumerate(conjunto_treinamento):
            # 1. Calcular saída desejada
            saida_desejada = self.definir_saida_desejada(entradas)
            
            # 2. Calcular saída obtida
            net = self.calcular_net(entradas)
            saida_obtida = self.funcao_ativacao(net)
            
            # 3. Calcular erro - Se for zero sabemos que nao tem erro
            erro_saida = saida_desejada - saida_obtida
            
            if erro_saida != 0:
                erro_epoca += 1
                
                # 4. Atualizar pesos para melhor convergencia do neuronio
                for j, _ in enumerate(self.pesos):
                    delta_w = self.taxa_aprendizado * erro_saida * entradas[j]
                    self.pesos[j] += delta_w
                
                # 5. Atualizar bias
                delta_bias = self.taxa_aprendizado * erro_saida
                self.bias += delta_bias
                
                self.interface_usuario.escrever("\n")
                self.interface_usuario.escrever(f"Padrão RU {i+1}: {''.join(map(str, entradas))}")
                self.interface_usuario.escrever(f"Net: {net:.3f}")
                self.interface_usuario.escrever(f"Saída desejada: {saida_desejada:+d}, Saída obtida: {saida_obtida:+d}")
                self.interface_usuario.escrever(f"Erro: {erro_saida:+d}")
                self.interface_usuario.escrever(f"Pesos atualizados: {[round(w, 3) for w in self.pesos]}")
                self.interface_usuario.escrever(f"Bias atualizado: {round(self.bias, 3)}")
                self.interface_usuario.escrever("\n")

            else:
                self.interface_usuario.escrever(f"Padrão (Item {i+1}) obteve convergencia ✅ : RU {''.join(map(str, entradas))}")
        
        return erro_epoca
    
    def treinar(self, conjunto_treinamento: list[list[float]], max_epocas=100):
        """
        Treina o perceptron até convergência ou máximo de épocas
        
        Args:
            conjunto_treinamento: lista de RU's com os padrões de dados
            (Ru superiores e inferiores)]

            max_epocas: Número máximo de épocas default. Como é um dataset simples
            a literatura indica 100 epocas de forma inicial
        """
        self.interface_usuario.escrever("Iniciando treinando....")
        self.interface_usuario.escrever(f"Conjunto de treinamento: {len(conjunto_treinamento)} padrões")
        self.interface_usuario.escrever("\n")
        
        for epoca in range(max_epocas):
            self.interface_usuario.escrever(f"ÉPOCA {epoca + 1}")
            erros_epoca = self.treinar_epoca(conjunto_treinamento)
            
            self.interface_usuario.escrever(f"Erros na época: {erros_epoca}")
            self.interface_usuario.escrever(f"Pesos finais da época: {[round(w, 3) for w in self.pesos]}") #Arrendonda para 3 casas decimais o float
            self.interface_usuario.escrever(f"Bias final da época: {round(self.bias, 3)}")
            self.interface_usuario.escrever("\n")
            
            if erros_epoca == 0:
                self.treinamento_sucesso = True
                self.interface_usuario.escrever(f"Convergencia alcançada na seguinte época : {epoca + 1}!")
                break

        if self.treinamento_sucesso is False:
            self.interface_usuario.escrever(f"Máximo de épocas ({max_epocas}) atingido sem convergência. Ajuste o bias e os pesos iniciais antes de realizar um novo treinamento")
        
        self.interface_usuario.escrever("Treinamento Finalizado")
    
    def testar(self, conjunto_teste : list[list[float]]):
        """
        Testa o perceptron com um conjunto de teste. 
        Importante : O conjunto de teste precisa ser amostras que diferem do conjunto de treinamento
        
        Args:
            conjunto_teste: lista de conjunto de testes
        """
        self.interface_usuario.escrever(f"Iniciando testes do conjunto de amostras contendo {len(conjunto_teste)} amostras")
        acertos = 0
        
        for i, entradas in enumerate(conjunto_teste):
            saida_desejada = self.definir_saida_desejada(entradas)
            saida_obtida = self.predizer(entradas)
            net = self.calcular_net(entradas)
            
            acertou = saida_desejada == saida_obtida
            if acertou:
                acertos += 1
            
            self.interface_usuario.escrever(f"Teste {i+1}: {entradas}")
            self.interface_usuario.escrever(f"  Net: {net:.3f}")
            self.interface_usuario.escrever(f"  Desejado: {saida_desejada:+d}, Obtido: {saida_obtida:+d}")
            self.interface_usuario.escrever(f"  Resultado: {'✅ ACERTO' if acertou else '❌ ERRO'}")
            self.interface_usuario.escrever("\n")
        
        # Estatistica basica para mostrar a porcentagem da precisão mediante a relação de quantidade de acertos com o total de elementos na amostra
        precisao_perceptron = (acertos / len(conjunto_teste)) * 100
        self.interface_usuario.escrever(f"\nPrecisão Perceptron: {acertos}/{len(conjunto_teste)} = {precisao_perceptron:.1f}%")


    def testar_assertividade(self):
        while True:

            if self.treinamento_sucesso is False:
                self.interface_usuario.escrever("\nNão é possivel continuar. Treinamento não iniciado ou não finalizado com sucesso ❌")
                break

            self.interface_usuario.escrever("\n")
            self.interface_usuario.escrever("Iniciando Teste Assertividade")
            self.interface_usuario.escrever(f"RU de Referência: {self.ru_referencia}")
            self.interface_usuario.escrever(f"Pesos Convergidos: {[round(w, 3) for w in self.pesos]}")

            ru_aluno_referencia = self.interface_usuario.ler("\nDigite um RU para validar o perceptron")
            ru_aluno_referencia_entradas = list(map(int, ru_aluno_referencia))

            if len(ru_aluno_referencia) != self.num_entradas:
                    self.interface_usuario.escrever(f"\n Númeto de entradas da RU informada não corresponde ao mesmo do treinamento: {self.num_entradas}")

            saida_esperada = self.definir_saida_desejada(ru_aluno_referencia_entradas)

            self.interface_usuario.escrever(f"\n Saida esperada do RU {ru_aluno_referencia}: {saida_esperada}")

            saida_obtida = self.predizer(ru_aluno_referencia_entradas)

            self.interface_usuario.escrever(f"  Resultado: {'✅ ACERTO' if saida_esperada == saida_obtida else '❌ ERRO'}")

            continuar = self.interface_usuario.ler("\n Deseja continuar ?. Y ou N")

            if continuar == "Y":
                continue

            break
