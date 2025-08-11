import random
import math

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
        
        print("=== PERCEPTRON INICIALIZADO ===")
        print(f"RU de Referência: {self.ru_referencia}")
        print(f"Pesos iniciais: {[round(w, 3) for w in self.pesos]}")
        print(f"Bias inicial: {self.bias}")
        print(f"Taxa de aprendizado: {self.taxa_aprendizado}")
        print()
    
    def funcao_ativacao(self, net):
        """
        Função de ativação degrau
        
        Args:
            net: Valor da entrada líquida
            
        Returns:
            +1 se net >= 0, -1 caso contrário
        """
        return 1 if net >= 0 else -1
    
    def calcular_net(self, entradas):
        """
        Calcula a entrada líquida (net)
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Valor da entrada líquida
        """
        net = sum(entrada * peso for entrada, peso in zip(entradas, self.pesos)) + self.bias
        return net
    
    def predizer(self, entradas):
        """
        Faz uma predição para um padrão de entrada
        
        Args:
            entradas: Lista com 7 dígitos
            
        Returns:
            Saída do neurônio (+1 ou -1)
        """
        net = self.calcular_net(entradas)
        return self.funcao_ativacao(net)
    
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
        todos_superiores = all(entrada > ref for entrada, ref in zip(entradas, self.ru_referencia))
        return 1 if todos_superiores else -1
    
    def treinar_epoca(self, conjunto_treinamento):
        """
        Treina o perceptron por uma época
        
        Args:
            conjunto_treinamento: Lista de padrões de entrada
            
        Returns:
            Número de erros na época
        """
        erros = 0
        
        for i, entradas in enumerate(conjunto_treinamento):
            # 1. Calcular saída desejada
            saida_desejada = self.definir_saida_desejada(entradas)
            
            # 2. Calcular saída obtida
            net = self.calcular_net(entradas)
            saida_obtida = self.funcao_ativacao(net)
            
            # 3. Calcular erro
            erro = saida_desejada - saida_obtida
            
            if erro != 0:
                erros += 1
                
                # 4. Atualizar pesos usando regra delta
                for j in range(len(self.pesos)):
                    delta_w = self.taxa_aprendizado * erro * entradas[j]
                    self.pesos[j] += delta_w
                
                # 5. Atualizar bias
                delta_bias = self.taxa_aprendizado * erro
                self.bias += delta_bias
                
                print(f"Padrão {i+1}: {entradas}")
                print(f"  Net: {net:.3f}")
                print(f"  Saída desejada: {saida_desejada:+d}, Saída obtida: {saida_obtida:+d}")
                print(f"  Erro: {erro:+d}")
                print(f"  Pesos atualizados: {[round(w, 3) for w in self.pesos]}")
                print(f"  Bias atualizado: {round(self.bias, 3)}")
                print()
        
        return erros
    
    def treinar(self, conjunto_treinamento, max_epocas=100):
        """
        Treina o perceptron até convergência ou máximo de épocas
        
        Args:
            conjunto_treinamento: Lista de padrões de entrada
            max_epocas: Número máximo de épocas
        """
        print("=== INICIANDO TREINAMENTO ===")
        print(f"Conjunto de treinamento: {len(conjunto_treinamento)} padrões")
        print()
        
        for epoca in range(max_epocas):
            print(f"--- ÉPOCA {epoca + 1} ---")
            erros = self.treinar_epoca(conjunto_treinamento)
            
            print(f"Erros na época: {erros}")
            print(f"Pesos finais da época: {[round(w, 3) for w in self.pesos]}")
            print(f"Bias final da época: {round(self.bias, 3)}")
            print()
            
            if erros == 0:
                print(f"🎉 CONVERGÊNCIA ALCANÇADA NA ÉPOCA {epoca + 1}!")
                break
        else:
            print(f"⚠️ Máximo de épocas ({max_epocas}) atingido sem convergência completa.")
        
        print("=== TREINAMENTO CONCLUÍDO ===")
    
    def testar(self, conjunto_teste):
        """
        Testa o perceptron com um conjunto de teste
        
        Args:
            conjunto_teste: Lista de padrões de teste
        """
        print("=== TESTE DO MODELO ===")
        acertos = 0
        
        for i, entradas in enumerate(conjunto_teste):
            saida_desejada = self.definir_saida_desejada(entradas)
            saida_obtida = self.predizer(entradas)
            net = self.calcular_net(entradas)
            
            acertou = saida_desejada == saida_obtida
            if acertou:
                acertos += 1
            
            print(f"Teste {i+1}: {entradas}")
            print(f"  Net: {net:.3f}")
            print(f"  Desejado: {saida_desejada:+d}, Obtido: {saida_obtida:+d}")
            print(f"  Resultado: {'✅ ACERTO' if acertou else '❌ ERRO'}")
            print()
        
        precisao = (acertos / len(conjunto_teste)) * 100
        print(f"Precisão: {acertos}/{len(conjunto_teste)} = {precisao:.1f}%")

def gerar_conjunto_treinamento():
    """
    Gera conjunto de treinamento com padrões variados
    """
    padroes = []
    
    # RU de referência
    ru_ref = [5, 1, 4, 5, 8, 7, 4]
    padroes.append(ru_ref)
    
    # Padrões superiores (todos os dígitos maiores)
    padroes.extend([
        [6, 2, 5, 6, 9, 8, 5],
        [7, 3, 6, 7, 9, 9, 6],
        [8, 4, 7, 8, 9, 9, 7],
        [9, 5, 8, 9, 9, 9, 8]
    ])
    
    # Padrões inferiores (pelo menos um dígito menor ou igual)
    padroes.extend([
        [4, 1, 4, 5, 8, 7, 4],  # Primeiro dígito menor
        [5, 0, 4, 5, 8, 7, 4],  # Segundo dígito menor
        [5, 1, 3, 5, 8, 7, 4],  # Terceiro dígito menor
        [1, 0, 0, 0, 0, 0, 0],  # Todos menores
        [2, 1, 3, 4, 5, 6, 3],  # Misturado
        [5, 1, 4, 5, 8, 7, 3],  # Último dígito menor
    ])
    
    return padroes

def gerar_conjunto_teste():
    """
    Gera conjunto de teste independente
    """
    return [
        [6, 3, 5, 6, 9, 8, 5],  # Superior
        [3, 0, 2, 3, 6, 5, 2],  # Inferior
        [9, 9, 9, 9, 9, 9, 9],  # Superior
        [1, 1, 1, 1, 1, 1, 1],  # Inferior
        [7, 4, 6, 8, 9, 8, 6],  # Superior
    ]

# Exemplo de uso
if __name__ == "__main__":
    # Criar e treinar o perceptron
    perceptron = PerceptronRU(taxa_aprendizado=0.1)
    
    # Gerar conjuntos de dados
    conjunto_treinamento = gerar_conjunto_treinamento()
    conjunto_teste = gerar_conjunto_teste()
    
    # Treinar
    perceptron.treinar(conjunto_treinamento, max_epocas=50)
    
    # Testar
    perceptron.testar(conjunto_teste)
    
    print("\n=== ANÁLISE FINAL ===")
    print(f"RU de referência: {perceptron.ru_referencia}")
    print(f"Pesos finais: {[round(w, 3) for w in perceptron.pesos]}")
    print(f"Bias final: {round(perceptron.bias, 3)}")
    
    # Teste manual com o próprio RU
    print(f"\nTeste com o próprio RU {perceptron.ru_referencia}:")
    resultado = perceptron.predizer(perceptron.ru_referencia)
    print(f"Resultado: {resultado:+d} (esperado: -1)")