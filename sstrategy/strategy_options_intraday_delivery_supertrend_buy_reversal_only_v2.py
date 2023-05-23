from datetime import time

import numpy as np
import pandas as pd
import talib

import clock
from segment import *
from strategy.core.strategy_options_base_v2 import StrategyOptionsBaseV2, OptionsInstrumentDirection, OptionsTradingsymbolSuffix, OptionsStrikeDirection
from strategy.utils import is_new_candle_generated, HistoricalDataCacher, check_order_complete_status, check_order_placed_successfully, data_sanity_check
from utils.func import check_argument, check_argument_bulk, is_nonnegative_int, is_positive_int, is_nonnegative_int_or_float

STRATEGY_KEY = 'STRATEGY_KEY'
HEDGE = 'HEDGE'
NORMAL = 'NORMAL'


class StrategyOptionsIntradayDeliverySupertrendReversalOnlyV2(StrategyOptionsBaseV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        VERSION = '3.3.1'
        CLIENT = 'SHANTANU V2'
        STRATEGY_TYPE = 'OPTIONS'
        self.logger.info(f'\n{"#" * 40}\nSTRATEGY VERSION: {VERSION}\n{"#" * 40}')
        self.logger.debug(f'\n{"#" * 60}\nSTRATEGY TYPE: {STRATEGY_TYPE} | CLIENT: {CLIENT}\n{"#" * 60}')

        parameter_string = '\n(1) FRESH_ORDER_CANDLE \n(1) START_TIME_HOURS \n(1) START_TIME_MINUTES \n(1) END_TIME_HOURS \n(1) END_TIME_MINUTES \n(1) SUPERTREND_FACTOR \n(1) SUPERTREND_LENGTH ' \
                           '\n(1) NO_OF_STRIKES_CE \n(1) NO_OF_STRIKES_PE \n(1) STRIKE_DIRECTION_CE \n(1) STRIKE_DIRECTION_PE \n(1) INSTRUMENT_SEGMENT \n(1) INSTRUMENT_EXCHANGE_TOKEN ' \
                           '\n(1) NO_OF_STRIKES_CE_HEDGE \n(1) NO_OF_STRIKES_PE_HEDGE \n(1) STRIKE_DIRECTION_CE_HEDGE \n(1) STRIKE_DIRECTION_PE_HEDGE \n(1) STOPLOSS_POINTS \n(1) HEDGE_POINTS \n(1) SL_RANGE'
        check_argument(self.strategy_parameters, 'extern_function', lambda x: len(x) >= 20, err_message=f'Need 20 parameters for this strategy: {parameter_string}')

        self.fresh_order_candle = self.strategy_parameters['FRESH_ORDER_CANDLE']
        self.start_time_hours = self.strategy_parameters['START_TIME_HOURS']
        self.start_time_minutes = self.strategy_parameters['START_TIME_MINUTES']
        self.end_time_hours = self.strategy_parameters['END_TIME_HOURS']
        self.end_time_minutes = self.strategy_parameters['END_TIME_MINUTES']
        self.supertrend_factor = self.strategy_parameters['SUPERTREND_FACTOR']
        self.supertrend_length = self.strategy_parameters['SUPERTREND_LENGTH']
        self.no_of_strikes_ce = self.strategy_parameters['NO_OF_STRIKES_CE']
        self.no_of_strikes_pe = self.strategy_parameters['NO_OF_STRIKES_PE']
        self._strike_direction_ce = self.strategy_parameters['STRIKE_DIRECTION_CE']
        self._strike_direction_pe = self.strategy_parameters['STRIKE_DIRECTION_PE']
        self.no_of_strikes_ce_hedge = self.strategy_parameters['NO_OF_STRIKES_CE_HEDGE']
        self.no_of_strikes_pe_hedge = self.strategy_parameters['NO_OF_STRIKES_PE_HEDGE']
        self._strike_direction_ce_hedge = self.strategy_parameters['STRIKE_DIRECTION_CE_HEDGE']
        self._strike_direction_pe_hedge = self.strategy_parameters['STRIKE_DIRECTION_PE_HEDGE']
        self.stoploss_points = self.strategy_parameters['STOPLOSS_POINTS']
        self.hedge_points = self.strategy_parameters['HEDGE_POINTS']
        self.instrument_exchange_token = self.strategy_parameters['INSTRUMENT_EXCHANGE_TOKEN']
        self.instrument_segment = self.strategy_parameters['INSTRUMENT_SEGMENT']
        self.sl_range = self.strategy_parameters['SL_RANGE']

        no_of_strikes_list = [(self.no_of_strikes_ce, 'NO_OF_STRIKES_CE'), (self.no_of_strikes_ce_hedge, 'NO_OF_STRIKES_CE_HEDGE'),
                              (self.no_of_strikes_pe, 'NO_OF_STRIKES_PE'), (self.no_of_strikes_pe_hedge, 'NO_OF_STRIKES_PE_HEDGE')]
        for no_of_strikes, txt in no_of_strikes_list:
            check_argument(no_of_strikes, 'extern_function', lambda x: 0 <= x <= 50 and isinstance(x, int), err_message=f'{txt} should be an integer with possible values between 0 to 50')

        strike_direction_list = [(self._strike_direction_ce, 'STRIKE_DIRECTION_CE'), (self._strike_direction_ce_hedge, 'STRIKE_DIRECTION_CE_HEDGE'),
                                 (self._strike_direction_pe, 'STRIKE_DIRECTION_PE'), (self._strike_direction_pe_hedge, 'STRIKE_DIRECTION_PE_HEDGE')]
        for strike_direction, txt in strike_direction_list:
            check_argument(strike_direction, 'extern_function', lambda x: x in [0, 1, 2] and isinstance(x, int), err_message=f'{txt} should be an integer with possible values - 0: ITM or 1: ATM or 2: OTM')

        is_nonnegative_int_arg_list = [self.start_time_hours, self.start_time_minutes, self.end_time_hours, self.end_time_minutes]
        is_positive_int_arg_list = [self.supertrend_length, self.fresh_order_candle, self.instrument_exchange_token, self.instrument_segment, self.stoploss_points, self.hedge_points, self.sl_range]
        is_nonnegative_int_or_float_arg_list = [self.supertrend_factor]

        check_argument_bulk(is_nonnegative_int_arg_list, 'extern_function', is_nonnegative_int, 'Value should be >=0')
        check_argument_bulk(is_positive_int_arg_list, 'extern_function', is_positive_int, 'Value should be >0')
        check_argument_bulk(is_nonnegative_int_or_float_arg_list, 'extern_function', is_nonnegative_int_or_float, 'Value should be >0.0')

        strike_direction_map = {0: OptionsStrikeDirection.ITM.value, 1: OptionsStrikeDirection.ATM.value, 2: OptionsStrikeDirection.OTM.value}

        self.strikes_direction_ce = strike_direction_map[self._strike_direction_ce]
        self.strikes_direction_pe = strike_direction_map[self._strike_direction_pe]
        self.strikes_direction_ce_hedge = strike_direction_map[self._strike_direction_ce_hedge]
        self.strikes_direction_pe_hedge = strike_direction_map[self._strike_direction_pe_hedge]

        segment_map = {1: SegmentNseEQ, 2: SegmentNseFO, 3: SegmentNseIDX, 4: SegmentMcxFO, 5: SegmentBseEQ, 6: SegmentBseFO, 7: SegmentNcdFO, 8: SegmentNyseEQ, 9: SegmentNasdaqEQ}

        segment, self.underlying_instrument = None, None

        exchange_token_to_tradingsymbol_map = {1035: 'INDIA VIX', 1001: 'NIFTY 50', 1016: 'NIFTY BANK', 1007: 'NIFTY FIN SERVICE'}

        try:
            segment = segment_map[self.instrument_segment]
            self.underlying_instrument = self.broker.get_instrument(segment=segment, tradingsymbol=exchange_token_to_tradingsymbol_map[self.instrument_exchange_token], master_data=True)

        except Exception as ex:
            self.logger.fatal(f'Could not create underlying instrument with segment {segment} and exchange token {self.instrument_exchange_token}, Kindly verify')
            raise SystemExit

        self.logger.debug(f"Underlying instrument {self.underlying_instrument}")

        try:
            self.candle_start_time = time(hour=self.start_time_hours, minute=self.start_time_minutes)
        except ValueError:
            self.logger.fatal('Error converting candle start time hours and minutes... EXITING')
            raise SystemExit

        try:
            self.candle_end_time = time(hour=self.end_time_hours, minute=self.end_time_minutes)
        except ValueError:
            self.logger.fatal('Error converting candle end time hours and minutes... EXITING')
            raise SystemExit

        self.hc = HistoricalDataCacher()
        self.KEY = f'VM_SOIDSBRO_V2_2023_{self.trading_type.value}_{self.strategy_mode.value}_{str(self.strategy_parameters)}_test_1'
        self.global_day_start = None
        self.hedge_main_order = None
        self.main_order = None
        self.memory_dictionary = None
        self.message_id = None
        self.reversal_flag = None
        self.stoploss_order = None
        self.strategy_dictionary = None
        self.tradingsymbol_suffix = None
        self.hedge_tradingsymbol_suffix = None
        self.yaml_instrument = None

        self.number_of_allowed_expiry_dates = 1

    @staticmethod
    def name():
        return 'Strategy Options Intraday Delivery Supertrend Buy Reversal Only V2'

    @staticmethod
    def versions_supported():
        return AlgoBullsEngineVersion.VERSION_3_3_0

    @staticmethod
    def day_start_and_resumed_order(instrument):
        latest_entered_order = None

        if instrument.orders:

            if len(instrument.orders) == 1:
                latest_entered_order = instrument.orders[0]

            elif len(instrument.orders) > 1:
                latest_entered_order = instrument.orders[-1]

        return latest_entered_order

    @staticmethod
    def get_exit_action(main_order):
        return ActionConstants.EXIT_BUY if main_order.order_transaction_type is BrokerOrderTransactionTypeConstants.BUY else ActionConstants.EXIT_SELL

    @staticmethod
    def supertrend(_df, period, multiplier, ohlc=('open', 'high', 'low', 'close')):

        def EMA(df, base, target, period, alpha=False):

            con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

            if alpha:
                # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
                df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
            else:
                # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
                df[target] = con.ewm(span=period, adjust=False).mean()

            df[target].fillna(0, inplace=True)
            return df

        def ATR(df, period, ohlc=('Open', 'High', 'Low', 'Close')):

            atr = 'atr'

            # Compute true range only if it is not computed and stored earlier in the df
            if 'TR' not in df.columns:
                df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
                df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
                df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

                df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

                df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

            # Compute EMA of true range using ATR formula after ignoring first row
            EMA(df, 'TR', atr, period, alpha=True)

            return df

        df = _df.copy()

        ATR(df, period, ohlc=ohlc)
        atr = 'atr'
        st = 'st'
        stx = 'stx'
        sma = 'sma'
        atr_wilder_series = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        df['atr'] = atr_wilder_series
        tr_series = talib.TRANGE(df['high'], df['low'], df['close'])
        atr_sma_series = talib.SMA(tr_series, timeperiod=period)
        df['sma'] = atr_sma_series
        # Compute basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df[sma]
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df[sma]

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i] else \
                    df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return df

    def initialize_instrument(self, instrument):
        super().initialize_instrument(self.underlying_instrument)

        if self.get_tradingsymbol_suffix_from_main_order(instrument=instrument) is None:
            self.yaml_instrument = instrument

    def initialize(self):
        super().initialize()
        self.global_day_start = True
        self.main_order = {}
        self.hedge_main_order = {}
        self.memory_dictionary = {}
        self.reversal_flag = {}
        self.stoploss_order = {}
        self.strategy_dictionary = {}
        self.tradingsymbol_suffix = {}
        self.hedge_tradingsymbol_suffix = {}

        number_of_allowed_expiry_dates = len(self.get_allowed_expiry_dates())
        if number_of_allowed_expiry_dates != self.number_of_allowed_expiry_dates:
            self.logger.info(f'Allowed expiry dates for this strategy are {self.number_of_allowed_expiry_dates}, got {number_of_allowed_expiry_dates}, exiting')
            raise SystemExit

        # Create memory dictionary
        if self.strategy_mode is StrategyMode.DELIVERY:
            _messages, success = self.get_messages(key=self.KEY)
            if not success:
                self.logger.error('Could not fetch messages!!!')
            print(_messages)
            if _messages:
                _message = _messages[0]
                self.memory_dictionary = _message['value']
                self.message_id = _message['id']
                self.logger.info(f'Fetched message: {_message} | {self.memory_dictionary} | {self.message_id}')
            else:
                self.logger.error(f'Message not found. Resetting! {_messages}')
                self.memory_dictionary = {}
                self.message_id = None

            # Sanity check
            if not isinstance(self.memory_dictionary, dict):
                self.logger.error('Fetched message is corrupt. Resetting!!!')
                self.memory_dictionary = {}
        else:
            self.memory_dictionary = {}
            self.message_id = None
        if self.message_id is None:
            self.logger.debug('Creating blank/dumb message dictionary for the first time')
            self.message_id, success = self.create_message(key=self.KEY, value=self.memory_dictionary)

    def get_decision(self, instrument, candle):
        tradingsymbol_suffix = None

        hist_data = self.hc.fetch_hist_data(instrument, candle, self.fresh_order_candle)

        if not data_sanity_check(hist_data):
            self.logger.debug(f'Hist data not found for {instrument} instrument - \n{hist_data}')
            return tradingsymbol_suffix

        supertrend_series = self.supertrend(hist_data, self.supertrend_length, self.supertrend_factor)['st']

        if not data_sanity_check(supertrend_series):
            self.logger.debug(f'Hist data not found for {instrument} instrument - \n{supertrend_series}')
            return tradingsymbol_suffix

        crossover = self.utils.crossover(hist_data['close'], supertrend_series, self.crossover_accuracy_decimals, 'CLOSE_SERIES', 'SUPERTREND_SERIES')

        crossover_map = {1: OptionsTradingsymbolSuffix.CE, -1: OptionsTradingsymbolSuffix.PE, 0: None}

        tradingsymbol_suffix = crossover_map[crossover]
        return tradingsymbol_suffix

    def get_hedge_decision(self, candle, tradingsymbol, main_order):
        ltp = self.broker.get_ltp(self.underlying_instrument)
        self.options_instruments_set_up_local(self.underlying_instrument, tradingsymbol, ltp)

        # Fetch local/child instrument on the basis of no of strikes and strike direction
        child_instrument = self.get_child_instrument_details(self.underlying_instrument, tradingsymbol, True)
        hist_data = self.hc.fetch_hist_data(main_order.instrument, candle, 1)

        if not data_sanity_check(hist_data):
            self.logger.debug(f'Hist data not found for {child_instrument} instrument - \n{hist_data}')
            return tradingsymbol, child_instrument

        hedge_points_series = [main_order.entry_price - self.hedge_points] * len(hist_data['close'])

        crossover = self.utils.crossover(hist_data['close'], hedge_points_series, self.crossover_accuracy_decimals, 'HEDGE_CLOSE_SERIES', 'HEDGE_POINTS_SERIES')

        if crossover == -1:
            return tradingsymbol, child_instrument
        return None, None

    def write_to_memory(self, message, _key, _data, hedge):
        base_inst_str = self.underlying_instrument.tradingsymbol
        self.strategy_dictionary[base_inst_str][hedge] = _data
        self.revise_memory_management(message)

    def read_from_memory(self):
        if self.memory_dictionary is not None:
            self.strategy_dictionary = self.memory_dictionary[STRATEGY_KEY]
        else:
            self.logger.debug(f'Failed to get info from memory, exiting')
            raise SystemExit

    def revise_memory_management(self, message=''):
        self.memory_dictionary[STRATEGY_KEY] = self.strategy_dictionary
        if self.strategy_mode is StrategyMode.DELIVERY:
            self.logger.info(message)
            _, success = self.update_message(self.message_id, self.KEY, self.memory_dictionary)
            if not success:
                self.logger.error('Could not update message')

    def strategy_select_instruments_for_entry(self, candle, instruments_bucket):
        selected_instruments_bucket = []
        sideband_info_bucket = []

        if clock.CLOCK.now().time() < self.candle_start_time:
            self.logger.debug(f'Start time not hit yet')
            return selected_instruments_bucket, sideband_info_bucket

        if clock.CLOCK.now().time() >= self.candle_end_time:
            self.logger.debug(f'End time hit')
            return selected_instruments_bucket, sideband_info_bucket

        for instrument in instruments_bucket:
            base_inst_str = self.underlying_instrument.tradingsymbol

            if self.strategy_dictionary.get(base_inst_str) is None:
                self.strategy_dictionary[base_inst_str] = {}

            if self.memory_dictionary.get(STRATEGY_KEY) is None:
                self.memory_dictionary[STRATEGY_KEY] = {}

            # YAML instrument: For entry/exit condition scanning, Underlying instrument: base_inst_str and to fetch child instrument, Child instrument: Main order placed on this
            # Check decision conditions only if decision is No Action or None i.e not reversal has not occurred
            if self.main_order.get(base_inst_str) is None and instrument.tradingsymbol == self.yaml_instrument.tradingsymbol and is_new_candle_generated(candle.start_datetime.time(), instrument.segment.get_official_trading_start_time(), self.fresh_order_candle):
                if self.tradingsymbol_suffix.get(base_inst_str) is None:
                    self.tradingsymbol_suffix[base_inst_str] = self.get_decision(instrument, candle)

                if self.tradingsymbol_suffix[base_inst_str] is not None:
                    ltp = self.broker.get_ltp(self.underlying_instrument)

                    # Setup ATM. OTM, ITM using current ltp for the base instrument
                    self.logger.debug(f"Instruments set up instrument {instrument} tradingsymbol suffix {self.tradingsymbol_suffix}")
                    self.options_instruments_set_up_local(self.underlying_instrument, self.tradingsymbol_suffix[base_inst_str], ltp)

                    # Fetch local/child instrument on the basis of no of strikes and strike direction
                    child_instrument = self.get_child_instrument_details(self.underlying_instrument, self.tradingsymbol_suffix[base_inst_str], False)

                    # Add mapping with base instrument and the child instrument
                    self.instruments_mapper.add_mappings(instrument, child_instrument)

                    selected_instruments_bucket.append(child_instrument)
                    sideband_info_bucket.append({'action': ActionConstants.ENTRY_BUY, 'base_instrument': self.underlying_instrument, 'hedge_flag': False})

            if self.main_order.get(base_inst_str) is not None and check_order_complete_status(self.main_order.get(base_inst_str)) and self.hedge_main_order.get(base_inst_str) is None:
                tradingsymbol = OptionsTradingsymbolSuffix.CE if self.tradingsymbol_suffix.get(base_inst_str) is OptionsTradingsymbolSuffix.PE else OptionsTradingsymbolSuffix.PE
                tradginsymbol_suffix, child_instrument = self.get_hedge_decision(candle, tradingsymbol, self.main_order.get(base_inst_str))

                if tradginsymbol_suffix is not None and child_instrument is not None:
                    self.hedge_tradingsymbol_suffix[base_inst_str] = tradginsymbol_suffix
                    self.instruments_mapper.add_mappings(instrument, child_instrument)

                    selected_instruments_bucket.append(child_instrument)
                    sideband_info_bucket.append({'action': ActionConstants.ENTRY_BUY, 'base_instrument': self.underlying_instrument, 'hedge_flag': True})
            self.reversal_flag[base_inst_str] = False
        return selected_instruments_bucket, sideband_info_bucket

    def strategy_enter_position(self, candle, instrument, sideband_info):
        self.logger.debug(f"Child instrument {instrument}")
        _qty = self.number_of_lots * instrument.lot_size

        if sideband_info['action'] is ActionConstants.ENTRY_BUY:
            _order = self.broker.BuyOrderRegular(instrument=instrument, order_code=self.order_code, order_variety=BrokerOrderVarietyConstants.MARKET, quantity=_qty)

        elif sideband_info['action'] is ActionConstants.ENTRY_SELL:
            _order = self.broker.SellOrderRegular(instrument=instrument, order_code=self.order_code, order_variety=BrokerOrderVarietyConstants.MARKET, quantity=_qty)

        else:
            self.logger.error(f'Got invalid sideband info: {sideband_info}')
            raise NotImplementedError

        # Save order details only if order is placed
        if check_order_placed_successfully(_order):
            base_inst_str = self.underlying_instrument.tradingsymbol
            hedge_flag = sideband_info['hedge_flag']
            if not hedge_flag:
                self.main_order[base_inst_str] = _order
                hedge = NORMAL
            else:
                self.hedge_main_order[base_inst_str] = _order
                hedge = HEDGE

            # Add the child instrument into the memory, to be fetched and used next day
            self.write_to_memory(f'For {instrument}, saving instrument string to memory', STRATEGY_KEY, instrument.tradingsymbol, hedge)
        return _order

    def strategy_select_instruments_for_exit(self, candle, instruments_bucket):
        selected_instruments_bucket = []
        sideband_info_bucket = []
        for instrument in instruments_bucket:
            if self.instruments_mapper.is_child_instrument(instrument):
                base_instrument = self.instruments_mapper.get_base_instrument(instrument)
                base_inst_str = self.underlying_instrument.tradingsymbol
                main_order = self.main_order.get(base_inst_str)
                if main_order is None and self.strategy_mode is StrategyMode.DELIVERY and self.global_day_start is True:
                    self.global_day_start = False
                    main_order = self.day_start_and_resumed_order(instrument)
                    self.tradingsymbol_suffix[base_inst_str] = self.get_tradingsymbol_suffix_from_main_order(main_order)
                    self.read_from_memory()

                    self.logger.debug(f"HEDGE {self.strategy_dictionary[base_inst_str][HEDGE]} | NORMAL: {self.strategy_dictionary[base_inst_str][NORMAL]}")
                    if self.strategy_dictionary[base_inst_str][HEDGE] is not None and self.strategy_dictionary[base_inst_str][HEDGE] == main_order.instrument.tradingsymbol:
                        self.hedge_main_order[base_inst_str] = main_order
                    elif self.strategy_dictionary[base_inst_str][NORMAL] is not None and self.strategy_dictionary[base_inst_str][NORMAL] == main_order.instrument.tradingsymbol:
                        self.main_order[base_inst_str] = main_order

                if check_order_complete_status(self.main_order.get(base_inst_str)):
                    self.check_and_place_stoploss_order(self.main_order.get(base_inst_str))
                    stoploss_order = self.stoploss_order.get(base_inst_str)
                    if check_order_complete_status(stoploss_order):
                        self.logger.debug(f"Stoploss order is complete for {instrument}")
                        selected_instruments_bucket.append(self.main_order[base_inst_str].instrument)
                        sideband_info_bucket.append({'action': self.get_exit_action(self.main_order[base_inst_str]), 'hedge_flag': False})

                        # Remove the child instrument from the strategy dictionary once order exited
                        self.write_to_memory(f'For {instrument}, removing instrument string from memory', STRATEGY_KEY, None, NORMAL)

                if check_order_complete_status(self.hedge_main_order.get(base_inst_str)) and is_new_candle_generated(candle.start_datetime.time(), instrument.segment.get_official_trading_start_time(), self.fresh_order_candle):
                    if self.check_hedge_exit_condition(base_instrument, candle, self.hedge_main_order.get(base_inst_str)):
                        self.logger.debug(f"Exit condition hit for hedge order {self.hedge_main_order[base_inst_str].instrument}")
                        selected_instruments_bucket.append(self.hedge_main_order[base_inst_str].instrument)
                        sideband_info_bucket.append({'action': self.get_exit_action(self.hedge_main_order[base_inst_str]), 'hedge_flag': True})

                        # Remove the child instrument from the strategy dictionary once order exited
                        self.write_to_memory(f'For {instrument}, removing instrument string from memory', STRATEGY_KEY, None, HEDGE)

                if self.main_order.get(base_inst_str) is not None and self.hedge_main_order.get(base_inst_str) is not None and self.tradingsymbol_suffix.get(base_inst_str) is not None:
                    tradingsymbol_suffix = self.get_decision(base_instrument, candle)
                    if (self.tradingsymbol_suffix[base_inst_str] is OptionsTradingsymbolSuffix.CE and tradingsymbol_suffix is OptionsTradingsymbolSuffix.PE) or \
                            (self.tradingsymbol_suffix[base_inst_str] is OptionsTradingsymbolSuffix.PE and tradingsymbol_suffix is OptionsTradingsymbolSuffix.CE):
                        self.logger.debug(f"Reversal condition satisfied for {instrument}")
                        self.tradingsymbol_suffix[base_inst_str] = tradingsymbol_suffix
                        self.reversal_flag[base_inst_str] = True
                        main_orders_list = [(self.main_order.get(base_inst_str), NORMAL), (self.hedge_main_order.get(base_inst_str), HEDGE)]
                        for main_order, hedge_flag in main_orders_list:
                            selected_instruments_bucket.append(main_order.instrument)
                            sideband_info_bucket.append({'action': self.get_exit_action(main_order), 'base_instrument': self.underlying_instrument, 'hedge_flag': hedge_flag})

                            # Remove the child instrument from the strategy dictionary once order exited
                            self.write_to_memory(f'For {instrument}, removing instrument string from memory', STRATEGY_KEY, None, hedge_flag)
        return selected_instruments_bucket, sideband_info_bucket

    def strategy_exit_position(self, candle, instrument, sideband_info):
        if sideband_info['action'] in [ActionConstants.EXIT_BUY, ActionConstants.EXIT_SELL]:
            base_inst_str = self.underlying_instrument.tradingsymbol
            main_order_position_closed = False
            hedge_flag = sideband_info['hedge_flag']
            main_order = self.hedge_main_order.get(base_inst_str) if hedge_flag else self.main_order.get(base_inst_str)
            stoploss_order = self.stoploss_order.get(base_inst_str) if not hedge_flag else None

            if stoploss_order is not None:

                # If complete, main order position is closed
                if check_order_complete_status(stoploss_order):
                    self.logger.debug(f"Stoploss is completed")
                    main_order_position_closed = True

                # If not complete, main order position is not closed, and cancel target order
                else:
                    self.logger.debug(f"Stoploss not completed")
                    stoploss_order.cancel_order()

            if main_order is not None:

                # Main order is complete
                if check_order_complete_status(main_order):

                    # If not squared off with target/stoploss, exit position now
                    if not main_order_position_closed:
                        main_order.exit_position()

                # Main order is not complete, cancel it
                else:
                    main_order.cancel_order()

            # All done, now reset everything
            self.set_all_none(hedge_flag)
            return True
        return False

    def check_hedge_exit_condition(self, base_instrument, candle, hedge_main_order):
        hist_data = self.hc.fetch_hist_data(base_instrument, candle, 1)

        if not data_sanity_check(hist_data):
            self.logger.debug(f'Hist data not found for {base_instrument} instrument - \n{hist_data}')
            return False

        supertrend_series = self.supertrend(hist_data, self.supertrend_length, self.supertrend_factor)['st']
        crossover = self.utils.crossover(hist_data['close'], supertrend_series, self.crossover_accuracy_decimals, 'HEDGE_CLOSE_SERIES', 'HEDGE_SUPERTREND_SERIES')
        tradingsymbol_suffix = self.get_tradingsymbol_suffix_from_main_order(main_order=hedge_main_order)

        if (tradingsymbol_suffix is OptionsTradingsymbolSuffix.CE and crossover == -1) or \
                (tradingsymbol_suffix is OptionsTradingsymbolSuffix.PE and crossover == 1):
            return True
        return False

    def check_and_place_stoploss_order(self, main_order):
        base_inst_str = self.underlying_instrument.tradingsymbol
        stoploss_order = self.stoploss_order.get(base_inst_str)
        if stoploss_order is None:
            trigger_price = main_order.entry_price - self.stoploss_points
            price = trigger_price * (1 - self.sl_range / 100)
            self.logger.info(f'Placing stoploss order for {main_order.instrument}')
            self.stoploss_order[base_inst_str] = self.place_exit_order(main_order.instrument, trigger_price, price, main_order.quantity, main_order, BrokerOrderVarietyConstants.STOPLOSS_LIMIT)

    def place_exit_order(self, instrument, trigger_price, price, quantity, main_order, variety, try_again=False):
        self.logger.info(f'For Exit Order: instrument: {instrument} | trigger price: {trigger_price} | price: {price} | qty: {quantity} | variety: {variety}')
        if trigger_price <= 0:
            self.logger.info(f'Price calculation has gone below zero, setting price to minimum tick size')
            trigger_price = instrument.tick_size
        if price <= 0:
            self.logger.info(f'Price calculation has gone below zero, setting price to minimum tick size')
            price = instrument.tick_size
        _order = None
        if main_order.order_transaction_type is BrokerOrderTransactionTypeConstants.BUY:

            try:
                _order = self.broker.SellOrderRegular(instrument=instrument, order_code=self.order_code, order_variety=variety, price=price,
                                                      trigger_price=trigger_price, quantity=quantity, position=BrokerExistingOrderPositionConstants.EXIT, related_order=main_order)

                if _order.broker_order_id is None or _order.get_order_status() is BrokerOrderStatusConstants.value.REJECTED:
                    raise Exception

            except Exception as ex:

                if variety is not BrokerOrderVarietyConstants.MARKET and try_again:
                    self.logger.info(f'Error while placing Exit order, trying again with variety as market order')
                    self.logger.debug(f'Error while placing order, details are as follows:\n{ex}')
                    variety = BrokerOrderVarietyConstants.MARKET
                    _order = self.place_exit_order(instrument, trigger_price, price, quantity, main_order, variety)

        elif main_order.order_transaction_type is BrokerOrderTransactionTypeConstants.SELL:

            try:
                _order = self.broker.BuyOrderRegular(instrument=instrument, order_code=self.order_code, order_variety=variety, price=price,
                                                     trigger_price=trigger_price, quantity=quantity, position=BrokerExistingOrderPositionConstants.EXIT, related_order=main_order)

                if _order.broker_order_id is None or _order.get_order_status() is BrokerOrderStatusConstants.value.REJECTED:
                    raise Exception

            except Exception as ex:
                if variety is not BrokerOrderVarietyConstants.MARKET and try_again:
                    self.logger.info(f'Error while placing Exit order, trying again with variety as market order')
                    self.logger.debug(f'Error while placing order, details are as follows:\n{ex}')
                    variety = BrokerOrderVarietyConstants.MARKET
                    _order = self.place_exit_order(instrument, trigger_price, price, quantity, main_order, variety)

        else:
            raise NotImplementedError

        return _order

    # Get tradingsymbol suffix for the previous day order
    def get_tradingsymbol_suffix_from_main_order(self, main_order=None, instrument=None):
        child_instrument = main_order.instrument.tradingsymbol if main_order is not None else instrument.tradingsymbol
        if child_instrument.endswith('CE') or child_instrument.endswith('CE [LOCAL]'):
            return OptionsTradingsymbolSuffix.CE
        elif child_instrument.endswith('PE') or child_instrument.endswith('PE [LOCAL]'):
            return OptionsTradingsymbolSuffix.PE
        return None

    def set_all_none(self, hedge_flag):
        base_inst_str = self.underlying_instrument.tradingsymbol
        if hedge_flag:
            self.hedge_main_order.pop(base_inst_str, None)
        else:
            self.main_order.pop(base_inst_str, None)
            self.stoploss_order.pop(base_inst_str, None)
        if self.reversal_flag.get(base_inst_str) in [None, False]:
            self.tradingsymbol_suffix.pop(base_inst_str, None)

    def get_child_instrument_details(self, base_instrument, tradingsymbol_suffix, hedge_flag):
        expiry_date = self.get_allowed_expiry_dates()[0]
        number_of_strikes_map = {OptionsTradingsymbolSuffix.CE: self.no_of_strikes_ce_hedge if hedge_flag else self.no_of_strikes_ce,
                                 OptionsTradingsymbolSuffix.PE: self.no_of_strikes_pe_hedge if hedge_flag else self.no_of_strikes_pe}
        strike_direction_map = {OptionsTradingsymbolSuffix.CE: self.strikes_direction_ce_hedge if hedge_flag else self.strikes_direction_ce,
                                OptionsTradingsymbolSuffix.PE: self.strikes_direction_pe_hedge if hedge_flag else self.strikes_direction_pe}
        child_instrument = self.get_options_instrument_with_strike_direction(base_instrument, expiry_date, tradingsymbol_suffix.value,
                                                                             strike_direction_map[tradingsymbol_suffix], number_of_strikes_map[tradingsymbol_suffix])
        return child_instrument

    def options_instruments_set_up_local(self, base_instrument, tradingsymbol_suffix, current_close, direction=OptionsInstrumentDirection.EXACT):
        expiry_dates = self.get_allowed_expiry_dates()

        for expiry_date in expiry_dates:
            self.options_instruments_set_up(base_instrument, direction, expiry_date, tradingsymbol_suffix.value, current_close)
