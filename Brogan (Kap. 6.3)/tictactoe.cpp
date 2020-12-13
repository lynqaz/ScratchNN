#include "tictactoe.h"

std::vector<double> tictactoe::getboard() {
	std::vector<double> board;
	for (int i = 0; i < 9; ++i) {
		if (m_board[i] == 1) {
			board.push_back(10);
		}
		else if (m_board[i] == 2) {
			board.push_back(-10);
		}
		else {
			board.push_back(0);
		}
	}
	int cp = getcurrentplayer();
	board.push_back(((signed)cp * 2 - 3) * 10);
	return board;
};
std::vector<int> tictactoe::getoriginalboard() {
	return m_board;
}
std::vector<int> tictactoe::getpossiblemoves() {
	//empty square = 1, occupied = 0
	std::vector<int> possibleMoves(9, 0);
	for (int i = 0; i < 9; ++i) {
		possibleMoves[i] = (m_board[i] == 0) ? 1 : 0;
	}
	return possibleMoves;
};
bool tictactoe::move(int square) {
	//set square to currentplayer
	if (square < 10 and square >= 0  and m_board[square] == 0) {
		m_board[square] = m_currentPlayer;
		m_moves += 1;
		changeplayer();
		return 1;
	}
	std::cout << "this move is not allowed!\n\n";
	return 0;
};
int tictactoe::getwinner() {
	if (checkboard(m_board, 3, -m_currentPlayer + 3)) {
		return -m_currentPlayer + 3;
	}
	return 0;
}
double tictactoe::getscore() {
	if (checkboard(m_board, 3, 1)) {
		return 1;
	}
	else if (checkboard(m_board, 3, 2)) {
		return 0;
	}
	else {
		return 0.5;
	}
}
void tictactoe::reset() {
	m_board = { 0,0,0,0,0,0,0,0,0 };
	m_currentPlayer = 1;
	m_moves = 0;
};

void tictactoe::changeplayer() {
	m_currentPlayer = -m_currentPlayer + 3;
};

bool tictactoe::checkboard(std::vector<int> board, int rowlength, int player) {
	//cols and rows
	for (int a = 0; a < 3; ++a) {
		int verticalSum = 0;
		int horizontalSum = 0;
		for (int b = 0; b < 3; ++b) {
			//each square of player1 += 1 whereas player2's squares += 4
			verticalSum += (int)pow(board[3 * a + b], 2);
			horizontalSum += (int)pow(board[3 * b + a], 2);
		}
		if (verticalSum == rowlength * (int)pow(player, 2)) {

			return true;
		}
		else if (horizontalSum == rowlength * (int)pow(player, 2)) {

			return true;
		}
	}
	//diagonals
	for (int a = 2; a < 5; a += 2) {
		int sum = 0;
		for (int b = 0; b < 3; b++) {
			sum += (int)pow(board[b * a + 4 - a], 2);
		}
		if (sum == rowlength * (int)pow(player, 2)) {
			return true;
		}
	}
	//neither player has 3 in a row
	return false;
};
void tictactoe::printboard() {

	std::cout << "-----------------------------------\nTIC TAC TOE :\nmove number: " << m_moves << "\ncurrent player : " << m_currentPlayer << std::endl << std::endl;
	for (int a = 1; a < 10; ++a) {
		std::string out;
		if (m_board[a - 1] == 1) {
			out = "X";
		}
		else if (m_board[a - 1] == 2) {
			out = "O";
		}
		else {
			out = " ";
		}
		std::cout << out << " ";
		if (a % 3 == 0) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl << std::endl;
};
int tictactoe::getcurrentplayer() {
	return m_currentPlayer;
};
int tictactoe::getmoves() {
	return m_moves;
}
int tictactoe::randommove() {
	std::vector<int> posMoves, v;
	v = getpossiblemoves();
	for (int i = 0; i < 9; ++i) {
		if (v[i] == 1) { posMoves.push_back(i); }
	}
	return posMoves[rand() % (signed)posMoves.size()];
};
int tictactoe::bruteforce() {
	//copy board to be able to make moves without changing the state of the original board
	std::vector<int> board = m_board;
	//initializing bestmove and -score with bestscore being -2 if the maximizing Player and 2 if minimizing Player can move next;
	int bestmove, bestscore, score, depth = 8 - m_moves;
	bool isMaximizing, tie = true;
	if (m_currentPlayer == 1) { bestscore = -2; isMaximizing = true; }
	else { bestscore = 2; isMaximizing = false; }
	//go through each empty square and discover the value of the according move using the minmax algorithm
	for (int square = 0; square < 9; ++square) {
		if (board[square] == 0) {
			board[square] = m_currentPlayer;
			score = minimax(board, depth, !isMaximizing);
			board[square] = 0;
			if (isMaximizing) {
				if (score >= bestscore) {
					bestscore = score;
					bestmove = square;
					tie = false;
				}
			}
			else {
				if (score <= bestscore) {
					bestscore = score;
					bestmove = square;
					tie = false;
				}
			}
		}
	}
	if (tie) {
		return 0;
	}
	return bestmove;
}
int tictactoe::minimax(std::vector<int> board, int depth, bool isMaximizing) {
	int bestscore = (isMaximizing) ? -2 : 2, score;
	//check if the last player has three in a row
	if (isMaximizing && checkboard(board, 3, 2)) {
		return -1;
	}
	else if (!isMaximizing && checkboard(board, 3, 1)) {
		return 1;
	}
	if (depth > 0) {
		for (int square = 0; square < 9; ++square) {
			if (board[square] == 0) {
				board[square] = (isMaximizing) ? 1 : 2;
				score = minimax(board, depth - 1, !isMaximizing);
				board[square] = 0;
				if (isMaximizing) {
					if (score > bestscore) {
						bestscore = score;
					}
				}
				else {
					if (score < bestscore) {
						bestscore = score;
					}
				}
			}
		}
	}
	else {
		return 0;
	}
	return bestscore;
}