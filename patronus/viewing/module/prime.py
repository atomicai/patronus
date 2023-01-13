import plotly.graph_objs as go
from wordcloud import WordCloud


def plotly_wordcloud(text, scale=1):
    wc = WordCloud(max_words=100, font_step=0.5, max_font_size=15, width=1250, height=450, scale=scale)
    wc.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 42 * 1)
    new_freq_list

    trace = go.Scatter(
        x=x,
        y=y,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo='text',
        hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
        mode='text',
        text=word_list,
    )

    layout = go.Layout(
        {
            'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            "paper_bgcolor": 'rgba(0,0,0,0)',
            "plot_bgcolor": 'rgba(0,0,0,0)',
            "width": 1250,
            "height": 450,
        }
    )

    fig = go.Figure(data=[trace], layout=layout)

    return fig


__all__ = ["plotly_wordcloud"]
