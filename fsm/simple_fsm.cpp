#include <iostream>
#include <string>

enum class State {
    SmallM,
    SuperM,
    FireM,
    DeadM,
};


class Mario {
public:
    Mario() {};
    Mario(const Mario&) = delete;
    Mario& operator=(const Mario&) = delete;
    ~Mario() {};

    void EatMushrooms() {
        switch (state_)
        {
        case State::SmallM:
            state_ = State::SuperM;
            score_ += 1000;
            std::cout << "change to SuperMario, get 1000 score, now " << score_ << std::endl;
            break;
        case State::SuperM:
            state_ = State::FireM;
            score_ += 1000;
            std::cout << "change to FireMario, get 1000 score, now " << score_ << std::endl;
            break;
        case State::FireM:
            score_ += 1000;
            std::cout << "already FireMario, get 1000 score, now " << score_ << std::endl;
            break;
        case State::DeadM:
            score_ = 0;
            std::cout << "Dead Mario, do nothing" << std::endl;
            break;
        default:
            break;
        }
    }

    void MeetMonsters() {

    }

    void Attack() {

    }

private:
    State state_ = State::SmallM;
    int score_ = 0;
};


int main(int argc, char** argv)
{
    Mario mario;

    mario.EatMushrooms();
    mario.EatMushrooms();
    mario.EatMushrooms();
    mario.EatMushrooms();
    mario.EatMushrooms();
    mario.EatMushrooms();
}