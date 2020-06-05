#include <vector>
#include "lib/littletest.hpp"

#include "knn.h"

// Tests Ejercicio 1

LT_BEGIN_SUITE(TestOrderedArray)

void set_up() {}
void tear_down() {}

LT_END_SUITE(TestOrderedArray)
LT_BEGIN_TEST(TestOrderedArray, InsertarVacio)
    OrderedArray a(4);
    a.insert({10, 0});
    LT_CHECK_EQ(a.k_nearest, a.k_nearest);
LT_END_TEST(InsertarVacio)

// Ejecutar tests
LT_BEGIN_AUTO_TEST_ENV()
    AUTORUN_TESTS()
LT_END_AUTO_TEST_ENV()
