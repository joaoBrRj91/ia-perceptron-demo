class AmostrasServico:
    """
    Classe de serviço que manipula as amostras através de métodos para gerar
    conjunto de treinamento e teste como para gerar o arquivo final
    em excel com todo o treinamento e resultado do perceptrom
    """
    def gerar_conjunto_treinamento(self):
        """
        Gera conjunto de treinamento com padrões variados

        Retorna 50 padrões fixos para treinamento do perceptron
        Classe 1 -> Todos >= ru_ref
        Classe 0 -> Pelo menos um valor < ru_ref
        """

        padroes = []
        
        # RU de referência
        ru_ref = [5, 1, 4, 5, 8, 7, 4]
        padroes.append(ru_ref)
        
        # Classe 1 (25 padrões)
        padroes.extend([
            [7, 3, 9, 9, 9, 9, 9],
            [7, 4, 7, 9, 9, 8, 9],
            [8, 3, 7, 7, 9, 8, 9],
            [8, 4, 8, 9, 9, 9, 9],
            [8, 4, 9, 9, 9, 8, 8],
            [6, 9, 7, 8, 9, 8, 7],
            [9, 4, 9, 8, 9, 9, 7],
            [8, 5, 8, 7, 9, 8, 7],
            [6, 4, 6, 8, 9, 9, 9],
            [8, 4, 6, 7, 9, 9, 6],
            [9, 4, 9, 7, 9, 8, 8],
            [8, 6, 9, 9, 9, 9, 6],
            [8, 9, 8, 8, 9, 9, 8],
            [7, 6, 8, 7, 9, 9, 7],
            [6, 3, 7, 9, 9, 8, 8],
            [9, 5, 6, 9, 9, 9, 9],
            [9, 3, 6, 8, 9, 8, 9],
            [7, 4, 6, 7, 9, 8, 6],
            [8, 5, 7, 8, 9, 8, 6],
            [9, 3, 7, 8, 9, 9, 6],
            [7, 3, 9, 7, 9, 9, 6],
            [7, 4, 9, 7, 9, 9, 9],
            [7, 9, 9, 8, 9, 9, 6],
            [9, 5, 7, 9, 9, 9, 9],
            [9, 4, 8, 7, 9, 8, 6]
        ])

        # Classe 0 (25 padrões)
        padroes.extend([
            [7, 1, 8, 7, 9, 8, 7],
            [7, 3, 9, 9, 9, 8, 9],
            [8, 4, 7, 9, 9, 9, 9],
            [6, 9, 7, 8, 9, 8, 7],
            [8, 5, 8, 7, 9, 8, 7],
            [6, 4, 6, 8, 9, 9, 9],
            [9, 4, 9, 7, 9, 8, 8],
            [8, 6, 9, 9, 9, 9, 6],
            [8, 9, 8, 8, 9, 9, 8],
            [7, 6, 8, 7, 9, 9, 7],
            [6, 3, 7, 9, 9, 8, 8],
            [9, 5, 6, 9, 9, 9, 9],
            [9, 3, 6, 8, 9, 8, 9],
            [7, 4, 6, 7, 9, 8, 6],
            [8, 5, 7, 8, 9, 8, 6],
            [9, 3, 7, 8, 9, 9, 6],
            [7, 3, 9, 7, 9, 9, 6],
            [7, 4, 9, 7, 9, 9, 9],
            [7, 9, 9, 8, 9, 9, 6],
            [9, 5, 7, 9, 9, 9, 9],
            [9, 4, 8, 7, 9, 8, 6],
            [6, 1, 9, 9, 9, 8, 9],
            [8, 4, 7, 9, 9, 9, 9],
            [6, 9, 7, 8, 9, 8, 7],
            [8, 5, 8, 7, 9, 8, 7]
        ])
        
        return padroes


    def gerar_conjunto_teste(self):
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