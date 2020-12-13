#pragma once
#include <vector>
#include <iostream>

class tictactoe
{
public:
	
	std::vector<double> getboard();
	std::vector<int> getpossiblemoves();
	int getcurrentplayer();
	int getwinner();
	int randommove();
	bool move(int);
	void printboard();
	void reset();
	int bruteforce();
	void randomboard();
	int getmoves();

private:

	std::vector<int> m_board = { 0,0,0,0,0,0,0,0,0, };
	int m_currentPlayer = 1;
	int m_moves = 0;

	void changeplayer();
	bool checkboard(std::vector<int>, int, int);
	int minimax(std::vector<int>, int, bool);
};

