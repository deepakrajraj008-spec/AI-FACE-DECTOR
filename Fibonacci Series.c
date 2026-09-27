#include <stdio.h>

void f1(int n, int c, int a, int b);

int main()
{
    int n;

    printf("enter value of n: ");
    scanf("%d", &n);

    f1(n, 1, -1, 1);

    return 0;
}

void f1(int n, int c, int a, int b)
{
    int s;

    if (c == n + 1)
    {
        printf(" ");
    }
    else
    {
        s = a + b;
        printf("\n%d", s);
        f1(n, c+1,b, s);
        }
        }