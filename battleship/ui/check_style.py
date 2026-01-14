from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.ui.theme import Theme


class CheckboxRadioStyle(QtWidgets.QProxyStyle):
    def drawPrimitive(self, element, option, painter, widget=None):
        if element not in (QtWidgets.QStyle.PE_IndicatorCheckBox, QtWidgets.QStyle.PE_IndicatorRadioButton):
            return super().drawPrimitive(element, option, painter, widget)

        rect = option.rect
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        border = QtGui.QColor(Theme.TEXT_LABEL)
        fill = QtGui.QColor(Theme.BG_BUTTON)
        if option.state & QtWidgets.QStyle.State_On:
            fill = QtGui.QColor(Theme.TEXT_LABEL)
            border = QtGui.QColor(Theme.TEXT_MAIN)
        elif option.state & QtWidgets.QStyle.State_NoChange:
            fill = QtGui.QColor(Theme.TEXT_LABEL)
            border = QtGui.QColor(Theme.TEXT_MAIN)

        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(QtGui.QBrush(fill))

        if element == QtWidgets.QStyle.PE_IndicatorRadioButton:
            radius = rect.width() / 2.0
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 3, 3)

        # Draw checkmark / dot
        if option.state & QtWidgets.QStyle.State_On:
            painter.setPen(QtGui.QPen(QtGui.QColor(Theme.BG_DARK), 2))
            if element == QtWidgets.QStyle.PE_IndicatorRadioButton:
                inner = rect.adjusted(4, 4, -4, -4)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(Theme.BG_DARK)))
                painter.drawEllipse(inner)
            else:
                x1 = rect.left() + rect.width() * 0.25
                y1 = rect.top() + rect.height() * 0.55
                x2 = rect.left() + rect.width() * 0.45
                y2 = rect.top() + rect.height() * 0.75
                x3 = rect.left() + rect.width() * 0.78
                y3 = rect.top() + rect.height() * 0.3
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.drawLine(int(x2), int(y2), int(x3), int(y3))
        elif option.state & QtWidgets.QStyle.State_NoChange and element == QtWidgets.QStyle.PE_IndicatorCheckBox:
            painter.setPen(QtGui.QPen(QtGui.QColor(Theme.BG_DARK), 2))
            y = rect.center().y()
            painter.drawLine(rect.left() + 3, y, rect.right() - 3, y)

        painter.restore()
        return None
