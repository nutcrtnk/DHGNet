run_setting = dict(

    bosnian=dict(
        target_lang='bs',
        multi_label=False,
        dataset='tweet_bos',
        text_col='text',
        label_col='label',
    ),

    bengali=dict(
        target_lang='bn',
        multi_label=False,
        dataset='bn_news',
        text_col='text',
        label_col='label',
    ),

    tamil=dict(
        target_lang='ta',
        multi_label=False,
        dataset='ta_news',
        text_col='text',
        label_col='label',
    ),

    malayalam=dict(
        target_lang='ml',
        multi_label=False,
        dataset='ml_news',
        text_col='text',
        label_col='label',
    ),

    thai_w=dict(
        target_lang='th',
        multi_label=False,
        dataset='wongnai',
        text_col='review',
        label_col='rating',
    ),

    thai_t=dict(
        target_lang='th',
        multi_label=False,
        dataset='truevoice',
        text_col='texts',
        label_col='destination',
    ),

)

globals().update(run_setting)


def update(args):
    setting = run_setting[args.run]
    args.__dict__.update({k: v for k, v in setting.items()
                         if k not in args or getattr(args, k) is None})
    return args
