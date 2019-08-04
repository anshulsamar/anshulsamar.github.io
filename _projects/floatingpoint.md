---
layout: narrative
title: FP32 and Alien Abductions
author: Anshul Samar
date: 2017-10-01
mydate: Oct 2017
---

Fall quarter 2017-18 I was a TA for CS107 at Stanford in which
students learn C, memory management, binary/floating point, and lots
of other intro to systems concepts. It's a challenging course for
students and a really involving class to staff.

I wrote up this little story to help my students as they were learning
about IEEE 32 bit floats. Enjoy! Special thanks to our teachers for
that quarter (Julia and Chris), CS107 class
materials, Bryant o'hallaron text, and other floating point resources.

--------------

Aliens have abducted Julie and Chris. As tradition, the aliens decide
to put these humans to the test. They come up with a game.

The rules of the game work as follows:
1) Aliens think of a number - it can be a whole number, fraction,
irrational number, anything - and tell it to Julie.
2) Julie is allowed to say a series of 32 ones and zeros to Chris.
3) Chris' job is to guess that number.
4) If Chris guesses the number, Aliens let Julie and Chris go. If they
don't, they use our favorite instructors as their afternoon meal.

Julie says - "Wait a second! This is impossible. 32 ones and zeros can
at most account for 2^32 different values. There are an infinite
number of numbers. This game is rigged!"

The aliens come together and discuss and realize (duh) that Julie is
right. So they modify the game. Chris doesn't have to guess the exact
number, but he has to come reasonably close. They are feeling
particularly generous, so any good faith attempt will do and they will
release Chris and Julie back to Earth.

How do Chris and Julie do it? They need to come up with something
fast. They do, after all, have to get back in time for the Friday
midterm.

Julie and Chris get sent to a room to discuss, with nothing but pen,
paper, and a cool decimal to binary converter that Chris
made. Luckily, I got a top secret transcript of their conversation.

Julie: Let's forget about base 10. A number is a number after all. If
these crazy aliens tell me some crazy fractional number, I can convert
this to binary. For example, I can convert the number 8.75 into it's
binary representation 1000.11. This number represents 1*2^3 + 0*2^2 +
0*2^1 * 0*2^0 + 1*2^{-1} + 1*2^{-2} = 8.75. But what if the binary
representation of the number they choose has more than 32 digits? How
would I relay this to you?

Chris: What if we have 16 digits represent everything before the
"decimal point"  and the next 16 represent everything after?

Julie: But then, the largest number we could represent would be 16
ones followed by a dot and 16 more ones (65535.99...). And the
smallest would be 0.

Chris: Hm. What if we represent numbers in 2^exponent *
significand. Multiplying by 2^exponent allows us move this decimal
place as far to the right or left as we want, letting our numbers
become arbitrarily big and small. We use part of the 32 bits to encode
the exponent and part of the 32 bits to encode the actual number. We
can call this latter part the significand because it holds the most
significant bits of the number.

Julie: Not sure if I follow?

Chris: The only difference between numbers like .0010001, 0.010001,
0.10001, 1.0001, 10.001, 100.01, 1000.1, 10001, 100010, 1000100 is
where the decimal point is! All of these numbers can be expressed as
2^E * 1.0001. All you have to do is tell me what the E is and what is
the significand and we can represent so many numbers!

Julie: That's brilliant. Now we can express really large numbers and
really small ones too. We take the most significant bits of a number
and then simply move the decimal place around. All we need to do now
is decide on a shared set of rules that let us map 32 bits into the
world of decimals. We may not be able to cover their number exactly -
but we can at least get close. Hopefully, this is good enough for the
aliens.

Julie and Chris decide that they want their 32 bits to represent as
many numbers as possible and that they will use a part of it as
exponent and part of it to cover the significand. They know that the
aliens might ask Julie any number whatsoever - so it would be
important for their bit representation to allow for positive and
negative numbers too. They decide to make this the first bit. If there
is a 1, the number is negative, and if it is a 0, then it is
positive.

Now, how many bits to use to encode the exponent and how many to keep
track of the actual number (the significand)?

They know that Chris will be tested on how precisely he can recover
the original number and so want to give plenty of bits for the
significand. Also, Julie really doubts whether these aliens are that
smart and so thinks that 8 exponent bits will be enough and the aliens
won't ask for too large of a number.

Julie: Ok, we have a sign bit, 8 bits for the exponent, and 23 bits
for the significand. How do I go from a binary representation of my
number to representing it this way?

Chris: Ok great! How about we look at the number represented by 8 bits
of the exponent and call that number E. We can look at the number
represented by the 23 bits of significand and call it M. The number
this represents could be 2^E * M?

Julie: I think you are onto something!  Let's assume for simplicity
that my exponent can be any number from 1 to 254 (we can save 0 and
255 for special cases). Also, since all we are doing is moving the
decimal around, how about we drop the first "1" from M that way I
don't have to waste a bit for it. For example, if I need to represent
1101.1, I can represent it as 2^3 * 1.1011. What's the point of
storing the 1 if it is understood? Even if I have a number with mostly
0s, like 0.000000101, I can write this as 1.01 * 2^{-7}. I will just
store the part after the decimal in M to save us a bit.

Chris: Sahweeeet. This brings up another good point. If E goes from 1
to 254, in our current scheme, I won't be able to encode for numbers
that are smaller than 0 (i.e. exponents like -7 or -12). Let's bias
the exponent by 127. This way we can small numbers too. So now, we
look at the 8 bits of exponent and subtract 127. This way we can have
the smallest E be 1 - 127 and the largest E be 254 - 127.  This gives
us 2^-126 (shift decimal left 126 places) all the way to 2^127 (shift
decimal right 126 places).

Julie: Phew. Makes sense. Let's do this.

[The astute 107 students will note that Julie/Chris have yet to talk
about INF, NaN, or denormalized floats. This is true but yours truly
did not have enough time to write that part of the story]

Finally, the moment of truth. Julie and Chris get called back to the
atrium and aliens tell Julie the number 0.4.

Julie does the following steps:

1) Convert .4 to binary.
0.011001100110011001100110011001100110011
There are even more digits than this, but we don't really need to know
more b/c they are all about to get truncated.

2) But our mantissa can only take 23 bits (and it gets the 1. on the
left for free). We oughta take the most important bits here and will
unfortunately have to drop the rest off. Why? If we have to store the
base 10 number 250620 and we have to pick the most important digits,
we're going to pick the most significant ones and say well, 250620 is
really pretty close to 2506 * 10^2. This is the same intuition that is
in play here for binary.

This means that our mantissa must be
1.10011001100110011001100  |   1100110011
everything to the right of that straight line is going to be cut
off. We round because instead of cutting it off, we decide we should
go to the closest number. So:

3)  1.10011001100110011001101
Also she put a 1 there on the left, but remember we get that for
free.

4) Now she has to figure out the exponent. To get back to the original
number we need to multiply by 2^-2. This means my exponent bits should
be 125 (since we are going to subtract by the bias).

She creates this number and relays it to Chris!

The aliens are holding their breath. The moment of truth.

Chris separates out the binary, exponent, and significand. And
calculates: 2^{E-127}*M and says: 0.400000006?

The aliens are shocked. True, it wasn't .4, but it was remarkably
close. Embarrassed to have lost their lunch, they send Julie and Chris
back home, just in time for the midterm. 