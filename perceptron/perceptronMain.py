from src.perceptron.PerceptronRU import PerceptronRU
from src.UI.InterfaceUsuarioFactory import InterfaceUsuarioFactory


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

    #Inicializa a UI de interação com o User. Obs : Só está implementado a console UI
    interface_usuario_comunicacao = InterfaceUsuarioFactory().criar_interface_usuario()

    # Criar e treinar o perceptron
    perceptron = PerceptronRU(taxa_aprendizado=0.1, interface_usuario=interface_usuario_comunicacao)
    
    # Gerar conjuntos de dados
    conjunto_treinamento = gerar_conjunto_treinamento()
    conjunto_teste = gerar_conjunto_teste()
    
    # Treinar
    perceptron.treinar(conjunto_treinamento)
    
    # Testar
    perceptron.testar(conjunto_teste)
    
    interface_usuario_comunicacao.escrever("=== ANALISE FINAL ===")
    interface_usuario_comunicacao.escrever(f"RU do alino: {perceptron.ru_referencia}")
    interface_usuario_comunicacao.escrever(f"Pesos finais: {[round(w, 3) for w in perceptron.pesos]}")
    interface_usuario_comunicacao.escrever(f"Bias final: {round(perceptron.bias, 3)}")
    
    # Teste manual com o próprio RU
    interface_usuario_comunicacao.escrever(f"\nTeste com o próprio RU {perceptron.ru_referencia}:")
    resultado = perceptron.predizer(perceptron.ru_referencia)
    interface_usuario_comunicacao.escrever(f"Resultado: {resultado:+d} (esperado: 1)")