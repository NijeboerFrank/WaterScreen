import unittest
import recognize_digits as d

class TestFiles(unittest.TestCase):

    def test_1_and_0(self):
        self.assertEqual(d.getNumberFromImage("test_images/test1.jpg"), 410)
        self.assertEqual(d.getNumberFromImage("test_images/test3.jpg"), 154)

    def test_4_numbers(self):
        self.assertEqual(d.getNumberFromImage("test_images/test2.jpg"), 1503)
        self.assertEqual(d.getNumberFromImage("test_images/test4.jpg"), 1748)
        self.assertEqual(d.getNumberFromImage("test_images/test7.jpg"), 1907)

    def random_tests(self):
        self.assertEqual(d.getNumberFromImage("test_images/test5.jpg"), 981)
        self.assertEqual(d.getNumberFromImage("test_images/test6.jpg"), 664)
        self.assertEqual(d.getNumberFromImage("test_images/test8.jpg"), 362)




if __name__ == '__main__':
    unittest.main()