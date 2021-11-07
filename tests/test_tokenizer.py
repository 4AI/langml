# -*- coding: utf-8 -*-

import os
import pytest

from langml.tokenizer import Encoding, SPTokenizer, WPTokenizer


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


@pytest.mark.parametrize(
    "left_text,right_text",
    [
        (
            'I like apples', None,
        ),
        (
            '我喜欢吃苹果', None
        ),
        (
            'I like apples', '我喜欢吃苹果'
        ),
        (
            '你好呀\n👋', 'hello world!!!\n👋'
        ),
        (
            "My mom always said life was like a box of chocolates. You never know what you're gonna get.", None
        ),
        (
            '''This line irked me. It is from the movie ‘Forrest Gump’ where the protagonist, played by Tom Hanks, quotes his mother:\nForrest Gump: My momma always said, “Life was like a box of chocolates. You never know what you’re gonna get.”\nYet, you do know, don’t you? You are going to get a box of chocolates. Perhaps you won’t know the taste of the chocolate specifically, perhaps one is more minty, the other might have that little coffee-taste of sorts, but chocolates nonetheless.\nI thought about it yesterday in the shower and figured, sure, if you compare it like that everyone gets a box of chocolates. Some boxes are bigger, some are smaller. Some chocolate boxes probably only have one piece of chocolate in it if you are lucky, and women probably have 70% of the chocolate inside.\nUltimately though, you know what you are going to get: chocolate. I admit that if you are getting a box of chocolates gifted to you, yes, you will not know what is inside. Life is a gift after all, so it makes sense. Yet life in itself is not mysterious as such, it is portrayed as quite simple. We wander around looking for meaning behind the chocolate. What does the chocolate mean? What if my chocolate is bitter? I don’t like the dark chocolate, I like it a little lighter, or white, or with caramel, or nuts. What if I don’t like the chocolate?\nI was bothered by the idea that it could be so simple, yet I figure in the grand scheme of things, yes, life is like a box of chocolates. It is that simple. It appears mysterious, that box, on the outside. Who knows what kinds of chocolates are inside? Then you open it and see the forms and shapes of the various chocolates, but you don’t know how it tastes. Take a bite and find out! The mystery unravels itself in time, as you are living.\nThat’s life, taking a bite of that unknown piece of chocolate, but more than than simply taking a bite out of the chocolate and seeing how it tastes, but appreciating that taste. Not judging it for not being the taste you expected, or disappointed that it wasn’t as tasty as it looked. Nor, should it be the case, being sad it only has 5 pieces instead of 10, or that the box isn’t as big as you’d hoped. Enjoy the piece of chocolate, every piece, and don’t leave a single piece behind.\nLife is like a box of chocolates, so enjoy every bite.''', None  # NOQA
        ),
        (
            '''马尚龙 | 上海每一个区，各由一字来微缩 原创 马尚龙 大上海小龙弄\n\n一晃已经是7年前的事情——人一生通常也就是晃十几晃。 🈳️   \n   就是这么一晃之间，许多人和事晃没了，许多人和事晃出来了。比如，晃着晃着，闸北区没了；晃着晃着，临港新片区有了。\n\n承蒙上海航空传媒的抬爱，我担任了《上海魅力》这本书的主编。此书汇集了上海十七个区县的人文历史旅游地貌——2014年崇明尚未撤县，闸北仍是一区。当时我们给自己挖了一个坑，在每一个区县的图文组合之首，做一个“一字解读”，既要点睛，又要有趣和个性化，不求面面俱到，但求冰山一角，而后再用极短的文字解读这一个字。\n\n创意很好。只是所有的创意都是坑，坑挖好了，谁进这个坑呢？只有让“主编大人”跳进去吧。\n\n\n\n\n封面图片说明：伍蠡甫画，1964年 上海炼油厂一角\n\n\n于是我就写下了《上海魅力》十七个“上海之……”。为十七个区县创作浓缩版精华画作的是著名画家戴红倩。我和他的愉快合作始于《上海制造》。\n\n十七个区县的“上海之……”，纯属我个人的“瞎七搭八”，说不上历史依据，更没有官方意图。在当时的《上海航空》先期刊登后，竟也有人看到，还和我讨论。有说我写得妙的，也有不同意的，比如杨浦区“上海之重”，是否应该写成“上海之学”——大学多文化氛围浓。不过我就一意孤行，一个人“一”到底了。本身就是随意为之，一讨论一征求意见，好像是革命样板戏重点题材的创作套路了。\n\n松江：上海之根\n\n\n\n如果说泰晤士小镇是松江的时尚，如果说大学城是松江的心胸，那么广富林是上海的骄傲，当广富林被注水重现上海之初时，在水一方是上海的根。\n\n静安：上海之秀\n\n\n\n静是态度，安是祈福；最袖珍的公园里有最秀雅的梧桐，最拥挤的人群，无疑也是最秀慧的人文景观，一年一度的上海书展，一再创造着中国图书展览的纪录。\n\n嘉定：上海之速\n\n\n\n最极速的是F1，却不完全是，还有中国第一条高速公路沪嘉高速，还有第一个中德合资的汽车品牌大众桑塔纳，更有以此带来的嘉定速度。\n\n杨浦：上海之重\n\n\n\n重可以是工业，上海的重工业基地再次舍我其谁？重也可以是学业，上海的重点大学在此联袂；重还可以是历史，诸多重要的事件在此展开。\n\n金山：上海之焰\n\n\n\n海上升明月是唐诗的意境，海上升焰火是金山的节日，渔村的味道，啤酒节的喧闹，音乐节的声道……可以闲步，可以闲聊，可以闲雅。\n\n虹口：上海之灵\n\n\n\n民族的灵魂在这里发出呐喊，文坛大师的灵感至今回荡；这里传递着从邮政总局发出的信函，从甜爱路散发的爱情传说。\n\n崇明：上海之绿\n\n\n\n曾经的荒芜化为全上海规模最大的生态湿地，曾经的角落化为全上海空气最清新的天然氧吧，曾经的朝发夕至化为桥隧一路通。\n\n长宁：上海之虹\n\n\n\n冥冥中的命名，带来了冥冥中的使命，虹桥，是上海最早通向全世界的天虹之桥，也是通向宁静、通向文明、通向富裕的长虹之桥。\n\n徐汇：上海之影\n\n\n\n这里，有上海电影制片厂留下的光影；有藏书楼留下的文化的背影；有土山湾留下的历史的倒影；有徐家汇商圈留下的熙攘人群的斜影。\n\n宝山：上海之水\n\n\n\n东海、长江和黄浦江三水合一，是著名的“三夹水”，三夹水就在宝山的吴淞口回旋；三水在此汇合，却是以自己的颜色而区分彼此。海纳百川有了一个绝好的注释。如今的邮轮恰是被三夹水涌向远方。\n\n普陀：上海之玉\n\n\n\n玉佛寺外两条路的路名，不经意间体现了最高的禅意——南北向的江宁路，东西向的安远路——宁静而致远；或许，也是玉的境界吧。其实这又何尝不是上海人的生活向往？\n\n青浦：上海之泽\n\n\n\n洋洋大观淀山湖，小桥流水朱家角，尤其是引以为荣的崧泽文化，宏伟的篇幅可以展示，传奇的故事可以流传，浪漫的风情可以演绎——就因为是鱼米之乡和人文故地。奥特莱斯，福寿园，乃至扎肉大米，迥异而声名远扬。\n\n闸北：上海之魔\n\n\n\n曾经有过经典魔术，将两次淞沪战争时期被日军炸毁的火车站迅速修复；曾经有过古彩魔术，把滚地龙变成了新工房；曾经有过时尚魔术，马戏城成为上海旅游的重头戏；还将拉开未来魔术，闸北区融入了静安区。\n\n闵行：上海之宝\n\n\n\n闵行曾经离上海市中心极其遥远，在闵行上班的工人都不能每天回家，如今闵行本身就是居民云集。七宝，曾经久闻七宝大曲，却无法从容来去，如今的七宝古镇，也无法从容来去，只因游人为七宝而去。\n\n奉贤：上海之湾\n\n\n\n海湾是自然的地理态势，蓝天碧海，一览无余；海湾也是人工的创造，碧海金沙，是观海，也是嬉沙。海湾更是人文的港湾，“奉贤”二字，便是告知天下，社会贤达，在这一个港湾的地位。\n\n黄浦：上海之曦\n\n\n\n如果说，曦是每一天阳光之第一，那么，南京路就是第一，国际饭店就是第一，淮海路就是第一，中国人自己创办的自来水厂就是第一……许多历史深远的政治活动仍是第一，比如，一大会址。\n\n浦东：上海之珠\n\n\n\n最初，东方明珠只是一座电视塔，后来，它成为了旅游热地，再后来它已经不再是电视塔，不再是最高，但是它成为了浦东的同名词，成为了浦东的“代言人”，浦东就是东方之珠。\n\n\n\n《上海魅力》出版后，新的创意诞生，那就是后来的《爱上海》明信片珍藏版，仅600册，且由戴敦邦大师题“爱上海”。明信片出版时，崇明撤县改区，闸北并入静安，再没有十七个区县之说，所以《爱上海》是真正的绝版了。\n\n轮不到我画龙点睛，也挨不着我画蛇添足，我只是“大上海小龙弄”一番而已，浪花也不起的。\n\n本次公号文字太少了，就贴两段有关上海的文字吧。这是我写于2009年《海派格调》的段落——\n\n有很多事情，是否发生在上海就会有不同的结果；其他地域当然也会有类似的现象，但是不至于像上海一样，这样的事情会很多，效果会很强烈。尤其是当这样的事情很小，本不应该会有强烈的反响，在其他地域甚至就不会成为一件事情，在上海却能成为一个家喻户晓的新闻事件。一座桥的修整，一座烟囱的保留，可以反反复复成为饭桌上的谈资。\n\n\n\n上海的外白渡桥要做迁移式大修，原本这就是一个市政工程，苏州河上所有的桥都经历了修整，当然外白渡桥享受的是最特殊的待遇，迁移大修，而非拆旧造新。这一项市政工程从发布消息直至正式迁移，一直是上海文化界的热议话题，诸如地标式的意义，旧上海的门户，“华人与狗不得入内”的见证……最有意思的是，时尚界始终将它作为一个时尚事件，围绕外白渡桥的大修，在外白渡桥上举行一系列时尚活动，影星章子怡就在外白渡桥上拍过一组照片。还有人发散性思维地创意：将外白渡桥上拆下来的废旧铆钉嵌在水晶玻璃内，一定是很有意义的纪念品和装饰品，后来果然有了这份镶嵌了废铜烂铁的纪念品。是上海这座城市的意义，决定了一座老桥的地位。\n\n\n\n同样的意义也决定了一座烟囱的保留。在非常注重环境保护的上海市中心徐家汇绿地，不仅保留了小洋房，而且还保留了工业社会象征的烟囱。几十年前，这一座黑烟滚滚的烟囱是伟大的象征，小学生将它写进作文里，画进素描里；几十年后，它静静地矗立在绿地中，依然非常雄性，与他匹配的不再是粗犷的工厂，而是柔情的时尚，小孩子们在烟囱底下游戏拍照，“听妈妈讲过去的故事”。\n\n只有在烟囱已经属于“过去的故事”的时候，它的属性才会由工业演变为文化和时尚；假如这一个城市依然还饱受林立的烟囱和滚滚的黑烟，那么任何一座烟囱都是罪恶。是上海这一座城市的意义，改变了这一座烟囱的是非。\n\n马尚龙\n\n中国作家协会会员，上海作家协会理事、散文报告文学专业创作委员会副主任；编审\n\n民进上海市委出版传媒委员会副主任\n\n上海黄浦区明复图书馆理事长\n\n上海评弹团艺委会顾问\n\n著作主要分为三个系列，分别是《幽默应笑我》《与名人同窗》等杂文系列，《上海制造》《为什么是上海》《上海分寸》《上海女人》《上海路数》等上海系列，《卷手语》《有些意思你从来不懂》等随笔系列 。\n\n\n\n我的新书《上海分寸》，上海书店出版社2021年1月出版。\n\n欢迎关注我的微信公众号“大上海小龙弄”——\n\n原标题：《马尚龙 | 上海每一个区，各由一字来微缩》''', None  # NOQA
        )
    ]
)
def test_wordpiece_encode(left_text, right_text):
    # lowercase=True
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    raw_tokenizer = tokenizer.raw_tokenizer()
    ret1 = tokenizer.encode(left_text, right_text, return_array=False)
    ret2 = raw_tokenizer.encode(left_text, right_text)

    assert ret1.ids == ret2.ids
    assert ret1.segment_ids == ret2.type_ids
    assert ret1.tokens == ret2.tokens

    # lowercase=False
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=False)
    raw_tokenizer = tokenizer.raw_tokenizer()
    ret1 = tokenizer.encode(left_text, right_text, return_array=False)
    ret2 = raw_tokenizer.encode(left_text, right_text)

    assert ret1.ids == ret2.ids
    assert ret1.segment_ids == ret2.type_ids
    assert ret1.tokens == ret2.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            [('I like apples', '我喜欢吃苹果'), ('hello world!', '你好世界👋')],
            'post',
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 2769, 1599, 3614, 1391, 5741, 3362, 102],
                     [101, 8701, 8572, 106, 102, 872, 1962, 686, 4518, 100, 102, 0, 0]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '我', '喜', '欢', '吃', '苹', '果', '[SEP]'],
                        ['[CLS]', 'hello', 'world', '!', '[SEP]', '你', '好', '世', '界', '[UNK]', '[SEP]', '[PAD]', '[PAD]']]   
            ),
        ), (
            [('I like apples', '我喜欢吃苹果'), ('hello world!', '你好世界👋')],
            'pre',
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 2769, 1599, 3614, 1391, 5741, 3362, 102],
                     [0, 0, 101, 8701, 8572, 106, 102, 872, 1962, 686, 4518, 100, 102]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '我', '喜', '欢', '吃', '苹', '果', '[SEP]'],
                        ['[PAD]', '[PAD]', '[CLS]', 'hello', 'world', '!', '[SEP]', '你', '好', '世', '界', '[UNK]', '[SEP]']]
            )
        )
    ]
)
def test_wordpiece_batch_encode_padding(inputs, padding_strategy, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    pred = tokenizer.encode_batch(inputs, padding=True, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            ['I like apples', '我喜欢吃苹果', 'hello world!', '你好世界👋'],
            'post',
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 0, 0],
                     [101, 2769, 1599, 3614, 1391, 5741, 3362, 102],
                     [101, 8701, 8572, 106, 102, 0, 0, 0],
                     [101, 872, 1962, 686, 4518, 100, 102, 0]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '[PAD]', '[PAD]'],
                        ['[CLS]', '我', '喜', '欢', '吃', '苹', '果', '[SEP]'],
                        ['[CLS]', 'hello', 'world', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]'],
                        ['[CLS]', '你', '好', '世', '界', '[UNK]', '[SEP]', '[PAD]']]  
            ),
        ), (
            ['I like apples', '我喜欢吃苹果', 'hello world!', '你好世界👋'],
            'pre',
            Encoding(
                ids=[[0, 0, 101, 151, 8993, 8350, 8118, 102],
                     [101, 2769, 1599, 3614, 1391, 5741, 3362, 102],
                     [0, 0, 0, 101, 8701, 8572, 106, 102],
                     [0, 101, 872, 1962, 686, 4518, 100, 102]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]],
                tokens=[['[PAD]', '[PAD]', '[CLS]', 'i', 'like', 'apple', '##s', '[SEP]'],
                        ['[CLS]', '我', '喜', '欢', '吃', '苹', '果', '[SEP]'],
                        ['[PAD]', '[PAD]', '[PAD]', '[CLS]', 'hello', 'world', '!', '[SEP]'],
                        ['[PAD]', '[CLS]', '你', '好', '世', '界', '[UNK]', '[SEP]']]
            )
        )
    ]
)
def test_wordpiece_batch_encode_single(inputs, padding_strategy, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    pred = tokenizer.encode_batch(inputs, padding=True, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,expected',
    [
        (
            [('I like apples', '我喜欢吃苹果'), ('hello world!', '你好世界👋')],
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 2769, 1599, 3614, 1391, 5741, 3362, 102],
                     [101, 8701, 8572, 106, 102, 872, 1962, 686, 4518, 100, 102]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '我', '喜', '欢', '吃', '苹', '果', '[SEP]'],
                        ['[CLS]', 'hello', 'world', '!', '[SEP]', '你', '好', '世', '界', '[UNK]', '[SEP]']]   
            ),
        )
    ]
)
def test_wordpiece_batch_encode_no_padding(inputs, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    pred = tokenizer.encode_batch(inputs, padding=False, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            [('I like apples', '我喜欢吃苹果'), ('hello world!', '你好世界👋')],
            'post',
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 2769, 1599, 3614, 102],
                     [101, 8701, 8572, 106, 102, 872, 1962, 686, 4518, 102]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '我', '喜', '欢', '[SEP]'],
                        ['[CLS]', 'hello', 'world', '!', '[SEP]', '你', '好', '世', '界', '[SEP]']]   
            ),
        ),
        (
            [('I like apples', '我喜欢吃苹果'), ('hello world!', '你好世界👋')],
            'pre',
            Encoding(
                ids=[[101, 151, 8993, 8350, 8118, 102, 2769, 1599, 3614, 102],
                     [101, 8701, 8572, 106, 102, 872, 1962, 686, 4518, 102]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
                tokens=[['[CLS]', 'i', 'like', 'apple', '##s', '[SEP]', '我', '喜', '欢', '[SEP]'],
                        ['[CLS]', 'hello', 'world', '!', '[SEP]', '你', '好', '世', '界', '[SEP]']]   
            ),
        )
    ]
)
def test_wordpiece_batch_encode_truncation(inputs, padding_strategy, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    tokenizer.enable_truncation(max_length=10)
    pred = tokenizer.encode_batch(inputs, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    "test_input,skip_special_tokens,expected",
    [
        ([101, 872, 1962, 686, 4518, 100, 102], True, ['你', '好', '世', '界']),
        ([101, 872, 1962, 686, 4518, 100, 102], False, ['[CLS]', '你', '好', '世', '界', '[UNK]', '[SEP]']),
        ([101, 8701, 8572, 102], True, ['hello', 'world']),
        ([101, 8701, 8572, 102], False, ['[CLS]', 'hello', 'world', '[SEP]'])
    ]
)
def test_wordpiece_decode(test_input, skip_special_tokens, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)

    assert tokenizer.decode(test_input, skip_special_tokens=skip_special_tokens) == expected


@pytest.mark.parametrize(
    "left_text,right_text,expected",
    [
        (
            ''.join(['头'] + ['[UNK]'] * 2),
            ''.join(['头'] + ['[UNK]'] * 2),
            Encoding(
                ids=[101, 1928, 100, 100, 102, 1928, 100, 100, 102],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '头', '[UNK]', '[UNK]', '[SEP]', '头', '[UNK]', '[UNK]', '[SEP]'],
            )
        ),
        (
            ''.join(['头'] + ['[UNK]'] * 2),
            ''.join(['头'] + ['[UNK]'] * 8),
            Encoding(
                ids=[101, 1928, 100, 100, 102, 1928, 100, 100, 100, 102],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                tokens=['[CLS]', '头', '[UNK]', '[UNK]', '[SEP]', '头', '[UNK]', '[UNK]', '[UNK]', '[SEP]']
            )
        ),
        (
            ''.join(['头'] + ['[UNK]'] * 8),
            ''.join(['头'] + ['[UNK]'] * 2),
            Encoding(
                ids=[101, 1928, 100, 100, 100, 102, 1928, 100, 100, 102],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '头', '[UNK]', '[UNK]', '[UNK]', '[SEP]', '头', '[UNK]', '[UNK]', '[SEP]']
            )
        ),
        (
            ''.join(['头'] + ['[UNK]'] * 8),
            ''.join(['头'] + ['[UNK]'] * 8),
            Encoding(
                ids=[101, 1928, 100, 100, 100, 102, 1928, 100, 100, 102],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '头', '[UNK]', '[UNK]', '[UNK]', '[SEP]', '头', '[UNK]', '[UNK]', '[SEP]'],
            )
        ),
    ]
)
def test_wordpiece_truncation_post(left_text, right_text, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    tokenizer.enable_truncation(10, strategy='post')

    ret = tokenizer.encode(left_text, right_text, return_array=False)
    assert ret.ids == expected.ids
    assert ret.segment_ids == expected.segment_ids
    assert ret.tokens == expected.tokens


@pytest.mark.parametrize(
    "left_text,right_text,expected",
    [
        (
            ''.join(['[UNK]'] * 2 + ['尾']),
            ''.join(['[UNK]'] * 2 + ['尾']),
            Encoding(
                ids=[101, 100, 100, 2227, 102, 100, 100, 2227, 102],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '[UNK]', '[UNK]', '尾', '[SEP]', '[UNK]', '[UNK]', '尾', '[SEP]'],
            )
        ),
        (
            ''.join(['[UNK]'] * 2 + ['尾']),
            ''.join(['[UNK]'] * 8 + ['尾']),
            Encoding(
                ids=[101, 100, 100, 2227, 102, 100, 100, 100, 2227, 102],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                tokens=['[CLS]', '[UNK]', '[UNK]', '尾', '[SEP]', '[UNK]', '[UNK]', '[UNK]', '尾', '[SEP]']
            )
        ),
        (
            ''.join(['[UNK]'] * 8 + ['尾']),
            ''.join(['[UNK]'] * 2 + ['尾']),
            Encoding(
                ids=[101, 100, 100, 100, 2227, 102, 100, 100, 2227, 102],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '[UNK]', '[UNK]', '[UNK]', '尾', '[SEP]', '[UNK]', '[UNK]', '尾', '[SEP]']
            )
        ),
        (
            ''.join(['[UNK]'] * 8 + ['尾']),
            ''.join(['[UNK]'] * 8 + ['尾']),
            Encoding(
                ids=[101, 100, 100, 100, 2227, 102, 100, 100, 2227, 102],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '[UNK]', '[UNK]', '[UNK]', '尾', '[SEP]', '[UNK]', '[UNK]', '尾', '[SEP]'],
            )
        ),
    ]
)
def test_wordpiece_truncation_pre(left_text, right_text, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    tokenizer.enable_truncation(10, strategy='pre')

    ret = tokenizer.encode(left_text, right_text, return_array=False)
    assert ret.ids == expected.ids
    assert ret.segment_ids == expected.segment_ids
    assert ret.tokens == expected.tokens


@pytest.mark.parametrize(
    'test_input,start_index,end_index,expected',
    [
        (
            'I like apples.', 3, 4, 'apples'
        ),
        (
            '我好菜啊!!', 1, 3, '我好菜'
        )
    ]
)
def test_wordpiece_tokens_mapping(test_input, start_index, end_index, expected):
    tokenizer = WPTokenizer(vocab_path=os.path.join(data_dir, 'wp_cn_vocab.txt'), lowercase=True)
    pred = tokenizer.encode(test_input)
    mapping = tokenizer.tokens_mapping(test_input, pred.tokens)
    assert test_input[mapping[start_index][0]: mapping[end_index][1]] == expected


@pytest.mark.parametrize(
    "left_text,right_text,expected",
    [
        (
            'I like apples', None,
            Encoding(
                [2, 31, 101, 4037, 18, 3],
                [0, 0, 0, 0, 0, 0],
                ['[CLS]', '▁i', '▁like', '▁apple', 's', '[SEP]']
            )
        ),
        (
            'hello world', 'hello world!!!\n👋',
            Encoding(
                [2, 10975, 126, 3, 10975, 126, 28116, 13, 1, 3],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                ['[CLS]', '▁hello', '▁world', '[SEP]', '▁hello', '▁world', '!!!', '▁', '👋', '[SEP]']
            )
        ),
    ]
)
def test_sentencepiece_encode(left_text, right_text, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'), lowercase=True)
    ret = tokenizer.encode(left_text, right_text, return_array=False)

    assert ret.ids == expected.ids
    assert ret.segment_ids == expected.segment_ids
    assert ret.tokens == expected.tokens


@pytest.mark.parametrize(
    "test_input,skip_special_tokens,expected",
    [
        ([2, 10975, 126, 3], True, ['▁hello', '▁world']),
        ([2, 10975, 126, 3], False, ['[CLS]', '▁hello', '▁world', '[SEP]']),
        ([2, 10975, 126, 28116, 13, 3], True, ['▁hello', '▁world', '!!!', '▁']),
        ([2, 10975, 126, 28116, 13, 3], False, ['[CLS]', '▁hello', '▁world', '!!!', '▁', '[SEP]'])
    ]
)
def test_sentencepiece_decode(test_input, skip_special_tokens, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    
    assert tokenizer.decode(test_input, skip_special_tokens=skip_special_tokens) == expected


@pytest.mark.parametrize(
    "left_text,right_text,expected",
    [
        (
            ' '.join(['head'] + ['unknown'] * 2),
            ' '.join(['head'] + ['unknown'] * 2),
            Encoding(
                ids=[2, 157, 2562, 2562, 3, 157, 2562, 2562, 3],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁head', '▁unknown', '▁unknown', '[SEP]', '▁head', '▁unknown', '▁unknown', '[SEP]'],
            )
        ),
        (
            ' '.join(['head'] + ['unknown'] * 2),
            ' '.join(['head'] + ['unknown'] * 8),
            Encoding(
                ids=[2, 157, 2562, 2562, 3, 157, 2562, 2562, 2562, 3],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                tokens=['[CLS]', '▁head', '▁unknown', '▁unknown', '[SEP]', '▁head', '▁unknown', '▁unknown', '▁unknown', '[SEP]']
            )
        ),
        (
            ' '.join(['head'] + ['unknown'] * 8),
            ' '.join(['head'] + ['unknown'] * 2),
            Encoding(
                ids=[2, 157, 2562, 2562, 2562, 3, 157, 2562, 2562, 3],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁head', '▁unknown', '▁unknown', '▁unknown', '[SEP]', '▁head', '▁unknown', '▁unknown', '[SEP]']
            )
        ),
        (
            ' '.join(['head'] + ['unknown'] * 8),
            ' '.join(['head'] + ['unknown'] * 8),
            Encoding(
                ids=[2, 157, 2562, 2562, 2562, 3, 157, 2562, 2562, 3],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁head', '▁unknown', '▁unknown', '▁unknown', '[SEP]', '▁head', '▁unknown', '▁unknown', '[SEP]'],
            )
        ),
    ]
)
def test_sentencepiece_truncation_post(left_text, right_text, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    tokenizer.enable_truncation(10, strategy='post')

    ret = tokenizer.encode(left_text, right_text, return_array=False)
    assert ret.ids == expected.ids
    assert ret.segment_ids == expected.segment_ids
    assert ret.tokens == expected.tokens


@pytest.mark.parametrize(
    "left_text,right_text,expected",
    [
        (
            ' '.join(['unknown'] * 2 + ['tail']),
            ' '.join(['unknown'] * 2 + ['tail']),
            Encoding(
                ids=[2, 2562, 2562, 3424, 3, 2562, 2562, 3424, 3],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁unknown', '▁unknown', '▁tail', '[SEP]', '▁unknown', '▁unknown', '▁tail', '[SEP]'],
            )
        ),
        (
            ' '.join(['unknown'] * 2 + ['tail']),
            ' '.join(['unknown'] * 8 + ['tail']),
            Encoding(
                ids=[2, 2562, 2562, 3424, 3, 2562, 2562, 2562, 3424, 3],
                segment_ids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                tokens=['[CLS]', '▁unknown', '▁unknown', '▁tail', '[SEP]', '▁unknown', '▁unknown', '▁unknown', '▁tail', '[SEP]']
            )
        ),
        (
            ' '.join(['unknown'] * 8 + ['tail']),
            ' '.join(['unknown'] * 2 + ['tail']),
            Encoding(
                ids=[2, 2562, 2562, 2562, 3424, 3, 2562, 2562, 3424, 3],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁unknown', '▁unknown', '▁unknown', '▁tail', '[SEP]', '▁unknown', '▁unknown', '▁tail', '[SEP]']
            )
        ),
        (
            ' '.join(['unknown'] * 8 + ['tail']),
            ' '.join(['unknown'] * 8 + ['tail']),
            Encoding(
                ids=[2, 2562, 2562, 2562, 3424, 3, 2562, 2562, 3424, 3],
                segment_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                tokens=['[CLS]', '▁unknown', '▁unknown', '▁unknown', '▁tail', '[SEP]', '▁unknown', '▁unknown', '▁tail', '[SEP]'],
            )
        ),
    ]
)
def test_sentencepiece_truncation_pre(left_text, right_text, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    tokenizer.enable_truncation(10, strategy='pre')

    ret = tokenizer.encode(left_text, right_text, return_array=False)
    assert ret.ids == expected.ids
    assert ret.segment_ids == expected.segment_ids
    assert ret.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            [('I like apples', 'I like watermelones'), ('hello world!', 'hello world')],
            'post',
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 18, 3, 13, 1, 101, 308, 21008, 2696, 3],
                     [2, 10975, 126, 187, 3, 10975, 126, 3, 0, 0, 0, 0, 0, 0]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', 's', '[SEP]', '▁', 'I', '▁like', '▁water', 'melo', 'nes', '[SEP]'],
                        ['[CLS]', '▁hello', '▁world', '!', '[SEP]', '▁hello', '▁world', '[SEP]', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]  
            ),
        ), (
            [('I like apples', 'I like watermelones'), ('hello world!', 'hello world')],
            'pre',
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 18, 3, 13, 1, 101, 308, 21008, 2696, 3],
                     [0, 0, 0, 0, 0, 0, 2, 10975, 126, 187, 3, 10975, 126, 3]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', 's', '[SEP]', '▁', 'I', '▁like', '▁water', 'melo', 'nes', '[SEP]'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '[CLS]', '▁hello', '▁world', '!', '[SEP]', '▁hello', '▁world', '[SEP]']]
            )
        )
    ]
)
def test_sentencepiece_batch_encode_padding(inputs, padding_strategy, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    pred = tokenizer.encode_batch(inputs, padding=True, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,expected',
    [
        (
            [('I like apples', 'I like watermelones'), ('hello world!', 'hello world')],
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 18, 3, 13, 1, 101, 308, 21008, 2696, 3],
                     [2, 10975, 126, 187, 3, 10975, 126, 3]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', 's', '[SEP]', '▁', 'I', '▁like', '▁water', 'melo', 'nes', '[SEP]'],
                        ['[CLS]', '▁hello', '▁world', '!', '[SEP]', '▁hello', '▁world', '[SEP]']]
            ),
        )
    ]
)
def test_sentencepiece_batch_encode_no_padding(inputs, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    pred = tokenizer.encode_batch(inputs, padding=False, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            [('I like apples', 'I like watermelones'), ('hello world!', 'hello world')],
            'post',
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 3, 13, 1, 101, 3],
                     [2, 10975, 126, 187, 3, 10975, 126, 3, 0, 0]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', '[SEP]', '▁', 'I', '▁like', '[SEP]'],
                        ['[CLS]', '▁hello', '▁world', '!', '[SEP]', '▁hello', '▁world', '[SEP]', '<pad>', '<pad>']]
            ),
        ),
        (
            [('I like apples', 'I like watermelones'), ('hello world!', 'hello world')],
            'pre',
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 3, 13, 1, 101, 3],
                     [0, 0, 2, 10975, 126, 187, 3, 10975, 126, 3]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', '[SEP]', '▁', 'I', '▁like', '[SEP]'],
                        ['<pad>', '<pad>', '[CLS]', '▁hello', '▁world', '!', '[SEP]', '▁hello', '▁world', '[SEP]']]
            ),
        )
    ]
)
def test_sentencepiece_batch_encode_truncation(inputs, padding_strategy, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    tokenizer.enable_truncation(max_length=10)
    pred = tokenizer.encode_batch(inputs, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'inputs,padding_strategy,expected',
    [
        (
            ['I like apples', 'I like watermelones', 'hello world!'],
            'post',
            Encoding(
                ids=[[2, 13, 1, 101, 4037, 18, 3, 0],
                     [2, 13, 1, 101, 308, 21008, 2696, 3],
                     [2, 10975, 126, 187, 3, 0, 0, 0]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]],
                tokens=[['[CLS]', '▁', 'I', '▁like', '▁apple', 's', '[SEP]', '<pad>'],
                        ['[CLS]', '▁', 'I', '▁like', '▁water', 'melo', 'nes', '[SEP]'],
                        ['[CLS]', '▁hello', '▁world', '!', '[SEP]', '<pad>', '<pad>', '<pad>']]
            ),
        ), (
            ['I like apples', 'I like watermelones', 'hello world!'],
            'pre',
            Encoding(
                ids=[[0, 2, 13, 1, 101, 4037, 18, 3],
                     [2, 13, 1, 101, 308, 21008, 2696, 3],
                     [0, 0, 0, 2, 10975, 126, 187, 3]],
                segment_ids=[[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]],
                tokens=[['<pad>', '[CLS]', '▁', 'I', '▁like', '▁apple', 's', '[SEP]'],
                        ['[CLS]', '▁', 'I', '▁like', '▁water', 'melo', 'nes', '[SEP]'],
                        ['<pad>', '<pad>', '<pad>', '[CLS]', '▁hello', '▁world', '!', '[SEP]']]
            )
        )
    ]
)
def test_sentencepiece_batch_encode_single(inputs, padding_strategy, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'))
    pred = tokenizer.encode_batch(inputs, padding=True, padding_strategy=padding_strategy, return_array=False)
    assert pred.ids == expected.ids
    assert pred.segment_ids == expected.segment_ids
    assert pred.tokens == expected.tokens


@pytest.mark.parametrize(
    'test_input,start_index,end_index,expected',
    [
        (
            'I like watermelons.', 3, 5, 'watermelons'
        ),
        (
            'Hello world!!!', 1, 1, 'Hello'
        )
    ]
)
def test_sentencepiece_tokens_mapping(test_input, start_index, end_index, expected):
    tokenizer = SPTokenizer(vocab_path=os.path.join(data_dir, 'albert_vocab/30k-clean.model'), lowercase=True)
    pred = tokenizer.encode(test_input)
    mapping = tokenizer.tokens_mapping(test_input, pred.tokens)
    assert test_input[mapping[start_index][0]: mapping[end_index][1]] == expected
