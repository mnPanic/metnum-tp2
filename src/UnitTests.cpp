#include <vector>
#include "lib/littletest.hpp"

#include "knn.h"

// para tests
std::ostream& operator<<(std::ostream& os, const neighbor n) { 
    os << "(" << n.dist << ", " << n.digit << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<neighbor> ns) {
    for(unsigned int i = 0; i < ns.size(); i++) {
        os << "[" << i << "] " << ns[i];
    }

    return os;
}

LT_BEGIN_SUITE(TestOrderedArray)

void set_up() {}
void tear_down() {}

LT_END_SUITE(TestOrderedArray)

LT_BEGIN_TEST(TestOrderedArray, InsertarVacio)
    OrderedArray a(4);
    a.insert({10, 0});
    std::vector<neighbor> want = {
        {10, 0},
        {INFINITY, 0},
        {INFINITY, 0},
        {INFINITY, 0},
    };
    LT_CHECK_EQ(a.k_nearest, want);
LT_END_TEST(InsertarVacio)

LT_BEGIN_TEST(TestOrderedArray, InsertarLlenando)
    OrderedArray a(4);
    a.insert({10, 0});
    a.insert({20, 1});
    a.insert({5, 9});
    a.insert({7, 7});
    std::vector<neighbor> want = {
        {5, 9},
        {7, 7},
        {10, 0},
        {20, 1},
    };
    LT_CHECK_EQ(a.k_nearest, want);
LT_END_TEST(InsertarLlenando)

LT_BEGIN_TEST(TestOrderedArray, InsertarNoEntra)
    OrderedArray a(4);
    a.insert({10, 0});
    a.insert({20, 1});
    a.insert({5, 9});
    a.insert({7, 7});

    // Insertamos uno que no entra
    a.insert({30, 7});
    std::vector<neighbor> want = {
        {5, 9},
        {7, 7},
        {10, 0},
        {20, 1},
    };
    LT_CHECK_EQ(a.k_nearest, want);
LT_END_TEST(InsertarNoEntra)

LT_BEGIN_TEST(TestOrderedArray, InsertarSwapPrincipio)
    OrderedArray a(4);
    a.insert({10, 0});
    a.insert({20, 1});
    a.insert({5, 9});
    a.insert({7, 7});

    // Insertamos uno que no entra
    a.insert({1, 7});
    std::vector<neighbor> want = {
        {1, 7},
        {5, 9},
        {7, 7},
        {10, 0},
    };
    LT_CHECK_EQ(a.k_nearest, want);
LT_END_TEST(InsertarSwapPrincipio)

LT_BEGIN_TEST(TestOrderedArray, InsertarSwapMedio)
    OrderedArray a(4);
    a.insert({10, 0});
    a.insert({20, 1});
    a.insert({5, 9});
    a.insert({7, 7});

    // Insertamos uno que no entra
    a.insert({8, 7});
    std::vector<neighbor> want = {
        {5, 9},
        {7, 7},
        {8, 7},
        {10, 0},
    };
    LT_CHECK_EQ(a.k_nearest, want);
LT_END_TEST(InsertarSwapMedio)

// Ejecutar tests
LT_BEGIN_AUTO_TEST_ENV()
    AUTORUN_TESTS()
LT_END_AUTO_TEST_ENV()
