# Conclusion

Gemini does add value, although most of the value comes from gpt-5.2

# Context

Most of my testing has been with gpt-5.2. Let's now see how it performs with claude and gemini. Let's use the baseline we have for V1B (gpt-5.2-low, using problems 247ef758:1,31f7f899:1,7c66cb00:1,136b0064:1,16de56c4:1,36a08778:1,1818057f:1,38007db0:2,bf45cf4b:1,b0039139:2,1ae2feb7:1,7ed72f31:2,b5ca7ac4:1):
- V1B: 1818057f,bf45cf4b,7c66cb00
- V1B: 1818057f,31f7f899,bf45cf4b,b0039139
- V1B: 1818057f,38007db0,136b0064
- V1B: 1818057f,1ae2feb7
- V1B: 1ae2feb7,1818057f,38007db0,31f7f899,247ef758
- V1B: 1818057f,247ef758,7c66cb00
- V1B: 1818057f,247ef758,bf45cf4b
- V1B: 1818057f,1ae2feb7,16de56c4
- V1B: 1818057f,31f7f899,136b0064
- V1B: 1818057f,247ef758
- V1B: 1818057f,247ef758,b0039139
- V1B: 1818057f,bf45cf4b

I'll run the same problems for claude-opus-4.5-thinking-4000:
- 1818057f
- 38007db0,1818057f
- NA
- NA
- 1818057f
- 38007db0

So, it seems to be somewhat underperforming, but then this is an arbitrary thinking level. At least, it runs without any major errors.

Now let's try gemini-3-low:
- 1818057f,38007db0,bf45cf4b,7ed72f31

Ok, both models are working. That's good.


# Is there even a point in using Opus and Gemini? Do they add something?

## "Impossible 9"
Let's run a few very hard but solvable problems (64efde09:1,7666fa5d:1,f560132c:1,de809cff:1,13e47133:2,13e47133:1,269e22fb:2,4e34c42c:2,a25697e4:1) on max reasoning for all three models and compare their performance:

gpt-5.2-xhigh:
- 269e22fb:2 : 0/8
- a25697e4:1 : 0/8
- f560132c:1 : 1/8
- 7666fa5d:1 : 4/8
- 64efde09:1 : 2/8
- 13e47133:2 : 0/8
- 4e34c42c:2 : 1/8
- 13e47133:1 : 0/8
- de809cff:1 : 1/8

Overall (72 total): 9 pass, 29 fail, 34 timeout

claude-opus-4.5-thinking-60000 (36 total): 7 fail, 29 timeout

gemini-3-high:
- 13e47133:1 : 1/4
- 13e47133:2 : 1/4

Overall (36 total): 2 pass, 33 fail, 1 missing

So, it seems that claude is pretty useless for the code generation.
7 fail, 29 timeout

## Frontier 47

Let's instead test on the frontier problems (3dc255db:1,8b7bacbf:2,0934a4d8:1,e3721c99:2,62593bfd:1,88e364bc:1,20a9e565:2,e3721c99:1,a25697e4:2,3a25b0d8:2,7b0280bc:1,eee78d87:1,269e22fb:1,4e34c42c:1,78332cb0:1,7b80bb43:1,89565ca0:1,a32d8b75:1,64efde09:1,7666fa5d:1,f560132c:1,de809cff:1,13e47133:2,13e47133:1,a25697e4:1,4e34c42c:2,269e22fb:2,faa9f03d:1,88bcf3b4:2,21897d95:2,2b83f449:1,446ef5d2:1,21897d95:1,16b78196:1,da515329:1,2d0172a1:2,3a25b0d8:1,4c416de3:1,4c7dc4dd:2,5545f144:1,6ffbe589:1,78332cb0:2,8b7bacbf:1,9bbf930d:1,abc82100:1,b9e38dc0:1,e12f9a14:2) and see how the models perform on them:
- claude-opus-4.5-thinking-60000: 0 solved, 56 timed out, 40 fail (2x attempts on 47 problems)
- gemini-3-high: 5 solved, 1 timed out, 182 fail (4x attempts on 47 problems)
- gpt-5.2-xhigh: 26 solved, 207 timed out, 143 fail (8x attempts on 47 problems)

Gemini might be adding a bit of value, but it's really all about gpt-5.2. On half of the problems that gemini solved, there also was a gpt-5.2 solution, so there is some additional value from Gemini.

The problem with Gemini is that it's not doing a deep enough search:
- gpt-5.2-xhigh, 1489.49s average, 2652.03s (95%), $0.9896 cost, 69364.6 tokens (output)
- gemini-3-high, 251.69s average, 306.91s (95%), $0.0472 cost, 2158.6 tokens
- claude-opus-4.5-thinking-60000, 572.23s average, 849.84s (95%), $0.9963 cost, 38197.6 tokens

We need to somehow force gemini to search much much deeper.

After changing the prompt and fixing some math on how tokens are used, this is the result:

Model                           Count     Avg (s)     95% (s)     Max (s)
gemini-3-high                   20        355.33      525.68      596.64

Model                           Avg Cost    Total Cost    % of Total   Avg Input  Avg Output Avg Cached
gemini-3-high_1_step_1_codegen  $0.3668     $1.8338          25.25%    10245.4    28856.2    0.0       
gemini-3-high_2_step_1_codegen  $0.3854     $1.9269          26.53%    9537.8     30525.8    0.0       
gemini-3-high_3_step_1_codegen  $0.3770     $1.8849          25.95%    9498.6     29831.4    0.0       
gemini-3-high_4_step_1_codegen  $0.3237     $1.6183          22.28%    10575.0    25208.4    0.0    
