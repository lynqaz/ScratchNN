#include "tictactoe.h"
#include "AI.h"


std::vector<int> extent = {10,9};
double learningRate = 0.001;
int batch = 100;
std::vector<std::vector<double>> boardPositions, bestMoves;
int iterations = 100;
int move;

int main() {
	tictactoe ttt;
	AI NN(extent);
	for (int i = 0; i < iterations; ++i) {
		ttt.randomboard();
		ttt.printboard();
		int moves = ttt.getmoves();
		for (int m = 0; m < 9 - moves; ++m) {
			if (m % 2 == 1) {
				if (ttt.getwinner() != 0) {
					break;
				}
				int bestmove = ttt.bruteforce();
				ttt.move(bestmove);
				bestMoves.push_back({});
				boardPositions.push_back(ttt.getboard());
				for (int a = 0; a < 9; ++a) {
					bestMoves.end()[-1].push_back((a == bestmove) ? 1.0 : 0.0);
				}
			}
			else {
				if (ttt.getwinner() != 0) {
					break;
				}
				int bestmove = ttt.bruteforce();
				ttt.move(bestmove);
				bestMoves.push_back({});
				boardPositions.push_back(ttt.getboard());
				for (int a = 0; a < 9; ++a) {
					bestMoves.end()[-1].push_back((a == bestmove) ? 1.0 : 0.0);
				}
			}
		}
		ttt.reset();
	}
	std::vector<std::vector<std::vector<double>>> training_data;
	training_data.push_back(boardPositions);
	training_data.push_back(bestMoves);
	for(int i = 0; i < 1; ++i){
		NN.train(training_data, boardPositions, bestMoves, 30000, batch, learningRate);
		std::cout << "mehr Training --> 1\nTraining beenden --> 0" << std::endl;
		int eingabe;
		std::cin >> eingabe; if (eingabe == 1) { --i; }
	}


	ttt.printboard();
	int neuralNetwork = 0, randomPlayer = 0, ties = 0;
	for (int s = 0; s < 1000; s++) {
		for (int i = 0; i < 9; ++i) {
			if (i % 2 == 0) {
				ttt.move(ttt.randommove());
			}
			else {
				std::vector<double> out = NN.run(ttt.getboard());
				std::vector<int> posMoves = ttt.getpossiblemoves();
				int bestMove = 4;
				double highestScore = 0;
				for (int i = 0; i < 9; ++i) {
					if (posMoves[i]) {
						if (out[i] >= highestScore) {
							bestMove = i;
							highestScore = out[i];
						}
					}
				}
				ttt.move(bestMove);
			}
			int winner = ttt.getwinner();
			if (winner != 0) {
				ttt.printboard();
				if (winner == 1) {
					neuralNetwork += 1;
				}
				else {
					randomPlayer += 1;
				}
				std::cout << "\nSPIELER " << winner << " GEWINNT!!!\n\n";
				break;
			}
		}
		if (ttt.getwinner() == 0) {
			ties += 1;
		}
		ttt.printboard();
		ttt.reset();
	}
	std::cout << "Neuronales Netzwerk gewann :			" << neuralNetwork << " mal" << std::endl;
	std::cout << "zufaellig spielender Gegner gewann :   " << randomPlayer << " mal" << std::endl;
	std::cout << "Unentschieden :						" << ties << std::endl;
	std::cin >> move;
	return 0;
}