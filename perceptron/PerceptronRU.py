import random
from src.UI.InterfaceUsuarioFactory import InterfaceUsuarioFactory

class PerceptronRU:
    def __init__(self, num_entradas=7, taxa_aprendizado=0.1):
        """
        Inicializa o Perceptron para identificação de RU
        
        Args:
            num_entradas: Número de entradas (7 dígitos do RU)
            taxa_aprendizado: Taxa de aprendizado (η)
        """
        self.num_entradas = num_entradas
        self.taxa_aprendizado = taxa_aprendizado
        
        # Inicialização aleatória dos pesos entre -0.5 e 0.5
        self.pesos = [random.uniform(-0.5, 0.5) for _ in range(num_entradas)]
        self.bias = 0  # Inicializa bias como 0
        
        # RU de referência
        self.ru_referencia = [5, 1, 4, 5, 8, 7, 4]
        
        # Inicializa a UI de interação com o User. Obs : Só está implementado a console
        self.interface_usuario = InterfaceUsuarioFactory().criar_interface_usuario()
        self.interface_usuario.escrever("=== PERCEPTRON INICIALIZADO ===")
        self.interface_usuario.escrever(f"RU de Referência: {self.ru_referencia}")
        self.interface_usuario.escrever(f"Pesos iniciais: {[round(w, 3) for w in self.pesos]}")
        self. interface_usuario.escrever(f"Bias inicial: {self.bias}")
        self.interface_usuario.escrever(f"Taxa de aprendizado: {self.taxa_aprendizado}")
        self.interface_usuario.escrever("\n")
    
    def funcao_ativacao(self, net):
        """
        Função de ativação degrau
        
        Args:
            net: Valor da entrada líquida
            
        Returns:
            +1 se net >= 0, -1 caso contrário
        """
        
    
    def calcular_net(self, entradas):
        """
        Calcula a entrada líquida (net)
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Valor da entrada líquida
        """
       
    
    def predizer(self, entradas):
        """
        Faz uma predição para um padrão de entrada
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Saída do neurônio (+1 ou -1)
        """
       
    
    def definir_saida_desejada(self, entradas):
        """
        Define a saída desejada baseada na regra:
        +1 se TODOS os dígitos forem superiores ao RU de referência
        -1 caso contrário
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Saída desejada (+1 ou -1)
        """
        # Verifica se TODOS os dígitos são superiores aos do RU de referência
       
    
    def treinar_epoca(self, conjunto_treinamento):
        """
        Treina o perceptron por uma época
        
        Args:
            conjunto_treinamento: Lista de padrões de entrada
            
        Returns:
            Número de erros na época
        """
        
    
    def treinar(self, conjunto_treinamento, max_epocas=100):
        """
        Treina o perceptron até convergência ou máximo de épocas
        
        Args:
            conjunto_treinamento: Lista de padrões de entrada
            max_epocas: Número máximo de épocas
        """
        
    
    def testar(self, conjunto_teste):
        """
        Testa o perceptron com um conjunto de teste
        
        Args:
            conjunto_teste: Lista de padrões de teste
        """
