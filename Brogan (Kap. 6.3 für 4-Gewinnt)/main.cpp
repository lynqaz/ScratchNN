#include "VierGewinnt.h"
#include "AI.h"


std::vector<int> extent = { 43,8,1 };
double learningRate = 10;
std::vector<std::vector<double>> boardPositions, bestMoves;
int move;
AI NN(extent);
VierGewinnt VG;
void selfplay(int);
int isMoveLegal(std::vector<int>, int);
int findBestMove(std::vector<int> board, bool training, double currentplayer, double& scoreReturn, std::vector<int> lastmoves1 = {}, std::vector<int> lastmoves2 = {});
int occurencesOfNumberInVector(int, std::vector<int>);
double sigmoid(double x) {
	return 1 / (1 + exp(-x * .2));
}

int main() {
	int neuralNetwork = 0, randomPlayer = 0, ties = 0;
	selfplay(1000);
	VG.reset();
	VG.printboard();
	double score;
	for (int s = 0; s < 100; s++) {
		for (int i = 0; i < 42; ++i) {
			if (i % 2 == 1) {
				move = VG.randommove();
				//std::cin >> move;
				VG.move(move);
			}
			else {
				VG.move(findBestMove(VG.getoriginalboard(), false, (double)VG.getcurrentplayer() - 1, score));
			}
			if (VG.getwinner() != 0) {
				if (VG.getwinner() == 1) {
					neuralNetwork += 1;
				}
				else {
					randomPlayer += 1;
				}
				std::cout << "\nPLAYER " << VG.getwinner() << " WINS GAME " << s << "\n\n";
				break;
			}
		}
		if (VG.getwinner() == 0) {
			ties += 1;
		}
		VG.printboard();
		VG.reset();
	}
	std::cout << "Neural Network wins :  " << neuralNetwork << std::endl;
	std::cout << "Random Player wins :   " << randomPlayer << std::endl;
	std::cout << "ties :                 " << ties << std::endl;
	std::cin >> move;
	return 0;
}

void selfplay(int matches) {
	//counts the number of occurences a particular move in the most recent 100 moves;
	int in = 0;
	int x = 0, o = 0;
	double match = 1;
	do {
		std::vector<int> lastmoves1, lastmoves2;
		for (int i = 0; i < matches; ++i) {
			std::vector<std::vector<double>> inputs = {};
			VG.reset();
			for (int move = 0; move < 42; ++move) {
				//check whether last move was a winning one
				if (VG.getwinner() != 0) {
					if (VG.getwinner() == 1) {
						x += 1;
					}
					else {
						o += 1;
					}
					break;
				}
				std::vector<int> board = VG.getoriginalboard();
				double score;
				//let the NN find the best move
				int m = findBestMove(board, false, (double)VG.getcurrentplayer() - 1, score, lastmoves1, lastmoves2);
				if (move % 2 == 0) {
					lastmoves1.push_back(m);
					if (lastmoves1.size() > 100) { lastmoves1.erase(lastmoves1.begin()); }
				}
				else {
					lastmoves2.push_back(m);
					if (lastmoves2.size() > 100) { lastmoves2.erase(lastmoves2.begin()); }
				}
				VG.move(m);
				std::vector<double> splitboard = VG.getboard();
				inputs.push_back(splitboard);
			}
			VG.printboard();
			double score = VG.getscore();
			NN.selfplay({ inputs }, { { score } }, learningRate);
			std::cout << "                                                   X : " << x << "     O : " << o << std::endl;
		}
		std::cin >> in;
	} while (in == 1);
}
int findBestMove(std::vector<int> board, bool training, double currentplayer, double& scoreReturn, std::vector<int> lastmoves1, std::vector<int> lastmoves2) {
	std::vector<std::vector<double>> outputs, inputs;
	//each possible move gets rated by the NN, depending on the Player will the highest or the lowest be the next move
	int nextmove;
	double bestscore = currentplayer;
	for (int move = 0; move < 7; ++move) {
		int column = isMoveLegal(board, move);
		if (column != -1) {
			board[column * 7 + move] = VG.getcurrentplayer();
			//create splitboard for NN
			std::vector<double> splitboard;
			for (int i = 0; i < 42; ++i) {
				if (board[i] == 1) {
					splitboard.push_back(10);
				}
				else if (board[i] == 2) {
					splitboard.push_back(-10);
				}
				else {
					splitboard.push_back(0);
				}
			}
			if (VG.getcurrentplayer() == 1) {
				splitboard.push_back(10);
			}
			else {
				splitboard.push_back(-10);
			}
			//run splitboard as input through NN and store output in outputs
			outputs.push_back(NN.run(splitboard));
			//occurencesOfNumberInVector makes sure every so often a new move will be explored, even though the NN predicts another move to be better
			if (training) {
				if (outputs[move][0] * (-sigmoid((double)occurencesOfNumberInVector(move, lastmoves1)) + 1.5) > bestscore && VG.getcurrentplayer() == 1) {
					bestscore = outputs[move][0] * (-sigmoid((double)occurencesOfNumberInVector(move, lastmoves1)) + 1.5);
					nextmove = move;
				}
				else if (outputs[move][0] * (sigmoid((double)occurencesOfNumberInVector(move, lastmoves2))) < bestscore && VG.getcurrentplayer() == 2) {
					bestscore = outputs[move][0] * sigmoid((double)occurencesOfNumberInVector(move, lastmoves2));
					nextmove = move;
				}
			}
			else {
				if (outputs[move][0] > bestscore && VG.getcurrentplayer() == 1) {
					bestscore = outputs[move][0];
					nextmove = move;
				}
				else if (outputs[move][0] < bestscore && VG.getcurrentplayer() == 2) {
					bestscore = outputs[move][0];
					nextmove = move;
				}
			}
			board[column * 7 + move] = 0;
		}
		else {
			//this move is not possible and therefore should not be picked
			outputs.push_back({ {0.5} });
		}
	}
	scoreReturn = bestscore;
	return nextmove;
}
int occurencesOfNumberInVector(int number, std::vector<int> vector) {
	int occurences = 0;
	for (int i = 0; i < (signed)vector.size(); ++i) {
		if (vector[i] == number) {
			occurences += 1;
		}
	}
	return occurences;
}

int isMoveLegal(std::vector<int> board, int move) {
	for (int col = 5; col >= 0; --col) {
		if (board[7 * col + move] == 0) {
			return col;
		}
	}
	return -1;
}