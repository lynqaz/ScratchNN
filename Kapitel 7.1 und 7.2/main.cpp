#include "AI.h"
#include "tictactoe.h"
int rain();
int main() {
	std::vector<int> extent = { 3,3,1 };
	std::vector<std::vector<double>> test_input, test_output, training_input, training_output;
	training_input = { {1,1,1}, { 1,1,0 }, { 1,0,1 }, { 1,0,0 }, { 0,1,1 }, { 0,1,0 }, { 0,0,0 }, {0,0,1}, {1,1,1} };
	training_output = { {0}, {1},{1},{0},{1},{0},{0},{0},{0} };
	test_input = { { 0,0,0 }, { 1,0,1 }, { 1,0,0 }, { 1,1,0}, {1,1,1} };
	test_output = { {0},{1},{0}, {1}, {0} };

	double learningrate = 0.01;
	int iterations = 1000000, batchsize = 4;

	AI func(extent);
	func.train(training_input, training_output, test_input, test_output, iterations, batchsize, learningrate);
	std::cout << "output : " << std::endl;
	
	for (auto out : test_input) {
		for (auto y : func.run(out)) {
			std::cout << y << std::endl;
		}
	}
	
	std::cout << "loss: " << func.getloss(test_input, test_output) << std::endl;
	std::cin.get();
	rain();
	return 0;
}

int rain() {
	std::vector<int> extent = { 20,32,18,9 };
	double learningRate = 0.001;
	int batch = 30;
	std::vector<std::vector<double>> boardPositions, bestMoves;
	int iterations = 100;
	int move;
	tictactoe ttt;
	AI NN(extent);
	for (int i = 0; i < iterations; ++i) {
		ttt.randomboard();
		ttt.printboard();
		int moves = ttt.getmoves();
		for (int m = 0; m < 9 - moves; ++m) {
			if (m % 2 == 1) {
				int bestmove = ttt.bruteforce();
				ttt.move(bestmove);
				bestMoves.push_back({});
				boardPositions.push_back(ttt.getboard());
				for (int a = 0; a < 9; ++a) {
					bestMoves.end()[-1].push_back((a == bestmove) ? 1.0 : 0.0);
				}
			}
			else {
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
	int a = 0;
	for (int i = 0; i < 1; ++i) {
		++a;
		NN.train(boardPositions, bestMoves, boardPositions, bestMoves, 1000, batch, learningRate / a);
		std::cout << "more --> 1\nstop --> 0";
		int eingabe;
		std::cin >> eingabe; if (eingabe == 1) { --i; }
	}


	ttt.printboard();
	for (int s = 0; s < 10; s++) {
		for (int i = 0; i < 9; ++i) {
			if (i % 2 == 0) {
				std::cout << "Enter your move : "; std::cin >> move;
				i -= (ttt.move(move)) ? 0 : 1;
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
				std::cout << "\nPLAYER " << winner << " WINS!!!\n\n";
				ttt.reset();
				i = -1;
			}
			ttt.printboard();
		}
		ttt.reset();
	}

	return 0;
}
